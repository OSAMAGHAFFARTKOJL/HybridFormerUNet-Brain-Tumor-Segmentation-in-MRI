import torch
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from model import HybridFormerUNet
from dataset import eda_visualization, get_loaders
from losses import HybridLoss
from train import train_model, plot_training_curves
from evaluate import evaluate_model, plot_evaluation_results, visualize_gradcam, visualize_attention_gates

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = HybridFormerUNet().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {total_params/1e6:.2f}M')

# EDA
eda_visualization()

# Loaders
train_loader, val_loader, test_loader = get_loaders()

# Training
EPOCHS = 5
LR = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
criterion = HybridLoss()

history, best_dice = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, EPOCHS)
plot_training_curves(history)
print(f'Best Validation Dice: {best_dice:.4f}')

# Evaluation
model.load_state_dict(torch.load('best_hybridformerunet.pth', map_location=device))
metrics_df, all_imgs_for_viz, auc_score, ap_score = evaluate_model(model, test_loader, device)
plot_evaluation_results(metrics_df, all_imgs_for_viz, auc_score, ap_score, np.concatenate([p for p in all_preds_prob]), np.concatenate([t for t in all_true]))  # Note: all_preds_prob and all_true need to be collected in evaluate_model if not returned

# Visualizations
visualize_gradcam(model, test_loader, device)

for imgs, _ in test_loader:
    sample_img = imgs[:1]
    break
visualize_attention_gates(model, sample_img, device)

print('PROJECT SUMMARY')
print(f'Architecture: HybridFormerUNet')
print(f'Best Validation Dice: {best_dice:.4f}')
print(f'Test Dice: {metrics_df["dice"].mean():.4f} ± {metrics_df["dice"].std():.4f}')
print(f'Test IoU: {metrics_df["iou"].mean():.4f} ± {metrics_df["iou"].std():.4f}')
print(f'AUC-ROC: {auc_score:.4f}')
print(f'Average Precision: {ap_score:.4f}')