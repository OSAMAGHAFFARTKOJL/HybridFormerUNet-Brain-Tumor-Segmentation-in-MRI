import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub

# Download dataset
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
DATA_DIR = os.path.join(path, 'kaggle_3m')

# Find patient folders
patient_folders = [f.path for f in os.scandir(DATA_DIR) if f.is_dir()]

all_images = []
all_masks = []

for folder in patient_folders:
    images = sorted(glob.glob(os.path.join(folder, "*.tif")))
    for img in images:
        if "_mask" in img:
            all_masks.append(img)
        else:
            all_images.append(img)

all_images = sorted(all_images)
all_masks = sorted(all_masks)

print(f"Total images: {len(all_images)}")
print(f"Total masks: {len(all_masks)}")

# EDA Visualization
def eda_visualization():
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('LGG MRI Dataset — Sample Visualization')

    indices = random.sample(range(len(all_images)), 5)
    for col, idx in enumerate(indices):
        img = cv2.imread(all_images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((256, 256, 3), dtype=np.uint8)
        
        mask = cv2.imread(all_masks[idx], 0) if os.path.exists(all_masks[idx]) else np.zeros((256,256), dtype=np.uint8)
        
        axes[0, col].imshow(img)
        axes[0, col].set_title(f'MRI Scan {idx}')
        axes[0, col].axis('off')
        
        overlay = img.copy()
        if mask.max() > 0:
            overlay[mask > 127] = [255, 50, 50]
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f'+ Tumor Mask')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig('eda_samples.png')
    plt.show()

    tumor_count = sum(1 for m in all_masks if cv2.imread(m, 0).max() > 0)
    print(f'Total samples: {len(all_images)}')
    print(f'With tumor: {tumor_count} ({100*tumor_count/len(all_images):.1f}%)')

# Transforms
IMG_SIZE = 256
BATCH_SIZE = 16

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),
    A.GridDistortion(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((256,256,3), np.uint8)
        
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else np.zeros((img.shape[0], img.shape[1]), np.uint8)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        
        return img.float(), mask.long()

# Data Split and Loaders
SEED = 42
pairs = list(zip(all_images, all_masks))
train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=SEED)
train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.15, random_state=SEED)

train_imgs, train_msks = zip(*train_pairs)
val_imgs, val_msks = zip(*val_pairs)
test_imgs, test_msks = zip(*test_pairs)

train_dataset = BrainMRIDataset(list(train_imgs), list(train_msks), train_transform)
val_dataset = BrainMRIDataset(list(val_imgs), list(val_msks), val_transform)
test_dataset = BrainMRIDataset(list(test_imgs), list(test_msks), val_transform)

def get_loaders(batch_size=BATCH_SIZE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader

print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}')