import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=5):
    scaler = torch.cuda.amp.GradScaler()
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': [], 'lr': []}
    best_dice = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred, ds_preds = model(imgs)
                loss = criterion(pred, ds_preds, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        scheduler.step()

        model.eval()
        val_loss, all_dice, all_iou = 0, [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast():
                    pred, ds_preds = model(imgs)
                    loss = criterion(pred, ds_preds, masks)
                val_loss += loss.item()
                pred_prob = F.softmax(pred, dim=1)[:, 1].cpu().numpy()
                true_np = masks.cpu().numpy()
                for p, t in zip(pred_prob, true_np):
                    m = compute_metrics(p, t)
                    all_dice.append(m['dice'])
                    all_iou.append(m['iou'])

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = np.mean(all_dice)
        avg_iou = np.mean(all_iou)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_dice)
        history['val_iou'].append(avg_iou)
        history['lr'].append(current_lr)

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), 'best_hybridformerunet.pth')

        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | LR: {current_lr:.2e}')

    return history, best_dice

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()

    axes[1].plot(epochs_range, history['val_dice'], label='Dice')
    axes[1].plot(epochs_range, history['val_iou'], label='IoU')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()

    axes[2].semilogy(epochs_range, history['lr'])
    axes[2].set_title('LR Schedule')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()