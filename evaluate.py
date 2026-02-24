import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from losses import compute_metrics

class SegGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        x = x.clone().detach().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            self.model.zero_grad()
            pred, _ = self.model(x)
            score = pred[:, 1].sum()
            score.backward()
            weights = self.gradients.mean(dim=(2,3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=True)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds_prob = []
    all_true = []
    all_imgs_for_viz = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs_gpu = imgs.to(device)
            pred, _ = model(imgs_gpu)
            prob = F.softmax(pred, dim=1)[:, 1].cpu().numpy()
            all_preds_prob.append(prob)
            all_true.append(masks.numpy())
            if len(all_imgs_for_viz) < 8:
                all_imgs_for_viz.extend([(imgs[i].numpy(), masks[i].numpy(), prob[i]) 
                                         for i in range(min(8 - len(all_imgs_for_viz), len(imgs)))])

    all_preds_prob = np.concatenate(all_preds_prob)
    all_true = np.concatenate(all_true)

    metrics_list = [compute_metrics(p, t) for p, t in zip(all_preds_prob, all_true)]
    metrics_df = pd.DataFrame(metrics_list)

    print('TEST SET EVALUATION RESULTS')
    for metric in ['dice', 'iou', 'precision', 'recall', 'f1']:
        vals = metrics_df[metric]
        print(f'{metric.capitalize()}: {vals.mean():.4f} ± {vals.std():.4f}')

    pred_binary = (all_preds_prob > 0.5).astype(int)
    flat_pred = pred_binary.flatten()
    flat_true = all_true.flatten()

    auc_score = roc_auc_score(flat_true, all_preds_prob.flatten())
    ap_score = average_precision_score(flat_true, all_preds_prob.flatten())
    print(f'AUC-ROC: {auc_score:.4f}')
    print(f'Avg Precision: {ap_score:.4f}')

    return metrics_df, all_imgs_for_viz, auc_score, ap_score

def plot_evaluation_results(metrics_df, all_imgs_for_viz, auc_score, ap_score, all_preds_prob, all_true):
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

    flat_true_sample = all_true.flatten()[:500000]
    flat_prob_sample = all_preds_prob.flatten()[:500000]

    # ROC
    ax_roc = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(flat_true_sample, flat_prob_sample)
    ax_roc.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    ax_roc.plot([0,1], [0,1], 'k--')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend()

    # PR
    ax_pr = fig.add_subplot(gs[0, 1])
    prec_c, rec_c, _ = precision_recall_curve(flat_true_sample, flat_prob_sample)
    ax_pr.plot(rec_c, prec_c, label=f'AP = {ap_score:.4f}')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend()

    # Metrics Bar
    ax_bar = fig.add_subplot(gs[0, 2])
    metric_names = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']
    metric_vals = [metrics_df[m].mean() for m in ['dice', 'iou', 'precision', 'recall', 'f1']]
    ax_bar.bar(metric_names, metric_vals)
    ax_bar.set_title('Segmentation Metrics')
    ax_bar.set_ylim(0, 1.1)

    # Dice Hist
    ax_hist = fig.add_subplot(gs[0, 3])
    ax_hist.hist(metrics_df['dice'], bins=20)
    ax_hist.axvline(metrics_df['dice'].mean(), color='red', linestyle='--')
    ax_hist.set_title('Dice Score Distribution')

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Sample Predictions
    for col, (img_np, mask_np, pred_prob) in enumerate(all_imgs_for_viz[:4]):
        ax = fig.add_subplot(gs[1, col])
        img_show = (img_np.transpose(1,2,0) * std + mean).clip(0,1)
        pred_bin = (pred_prob > 0.5).astype(np.uint8)
        overlay = img_show.copy()
        overlay[pred_bin == 1] = overlay[pred_bin == 1] * 0.5 + np.array([1,0,0]) * 0.5
        overlay[mask_np == 1] = overlay[mask_np == 1] * 0.5 + np.array([0,1,0]) * 0.5
        ax.imshow(overlay)
        m = compute_metrics(pred_prob, mask_np)
        ax.set_title(f'Dice={m["dice"]:.3f} IoU={m["iou"]:.3f}')
        ax.axis('off')

    # Probability Heatmaps
    for col, (img_np, mask_np, pred_prob) in enumerate(all_imgs_for_viz[:4]):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        ax.contour(mask_np, levels=[0.5], colors='cyan')
        ax.set_title('Probability Map + GT contour')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()

def visualize_gradcam(model, test_loader, device):
    cam_extractor = SegGradCAM(model, model.enc4[-1])

    for imgs, masks in test_loader:
        sample_imgs = imgs[:4]
        sample_masks = masks[:4]
        break

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    col_titles = ['Original MRI', 'Ground Truth', 'Prediction', 'GradCAM Heatmap', 'Overlay']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title)

    for row in range(4):
        img_tensor = sample_imgs[row:row+1].to(device)
        mask_np = sample_masks[row].numpy()
        cam = cam_extractor(img_tensor)
        with torch.no_grad():
            pred_logit, _ = model(img_tensor)
        pred_prob = F.softmax(pred_logit, dim=1)[0, 1].cpu().numpy()
        img_show = (sample_imgs[row].numpy().transpose(1,2,0) * std + mean).clip(0,1)

        axes[row, 0].imshow(img_show)
        axes[row, 1].imshow(mask_np, cmap='gray')
        axes[row, 2].imshow(pred_prob, cmap='hot', vmin=0, vmax=1)
        axes[row, 3].imshow(cam, cmap='jet')
        cam_colored = plt.cm.jet(cam)[:, :, :3]
        overlay = img_show * 0.6 + cam_colored * 0.4
        axes[row, 4].imshow(overlay.clip(0,1))

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_result.png')
    plt.show()

def visualize_attention_gates(model, sample_img, device):
    attention_maps = {}

    def get_attention_hook(name):
        def hook(module, input, output):
            avg = torch.mean(input[0], dim=1, keepdim=True)
            mx, _ = torch.max(input[0], dim=1, keepdim=True)
            attn = torch.sigmoid(module.conv(torch.cat([avg, mx], dim=1)))
            attention_maps[name] = attn.squeeze().detach().cpu().numpy()
        return hook

    hooks = [
        model.dec4.gate.sa.register_forward_hook(get_attention_hook('dec4')),
        model.dec3.gate.sa.register_forward_hook(get_attention_hook('dec3')),
        model.dec2.gate.sa.register_forward_hook(get_attention_hook('dec2')),
    ]

    model.eval()
    with torch.no_grad():
        _ = model(sample_img.to(device))

    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_show = (sample_img[0].numpy().transpose(1,2,0) * std + mean).clip(0,1)
    axes[0].imshow(img_show)
    axes[0].set_title('Input MRI')
    axes[0].axis('off')

    for i, (name, attn_map) in enumerate(attention_maps.items()):
        ax = axes[i+1]
        if attn_map.ndim > 2:
            attn_map = attn_map[0]
        im = ax.imshow(attn_map, cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Attention Gate: {name}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('attention_gates.png')
    plt.show()