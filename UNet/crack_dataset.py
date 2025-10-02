# crack_dataset.py
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=512):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks folder not found: {self.masks_dir}")

        imgs  = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in EXTS])
        masks = sorted([p for p in self.masks_dir.iterdir()  if p.suffix.lower() in EXTS])

        img_stems  = {p.stem: p for p in imgs}
        mask_stems = {p.stem: p for p in masks}
        common = sorted(set(img_stems.keys()) & set(mask_stems.keys()))

        self.pairs = [(img_stems[s], mask_stems[s]) for s in common]

        # helpful logs
        if len(self.pairs) == 0:
            missing_masks = sorted(set(img_stems) - set(mask_stems))
            missing_imgs  = sorted(set(mask_stems) - set(img_stems))
            raise RuntimeError(
                "No image/mask pairs found.\n"
                f"- Images dir: {self.images_dir}\n"
                f"- Masks dir : {self.masks_dir}\n"
                f"- Example image files: {[p.name for p in imgs[:5]]}\n"
                f"- Example mask files : {[p.name for p in masks[:5]]}\n"
                f"- Missing masks for images: {missing_masks[:10]}\n"
                f"- Missing images for masks: {missing_imgs[:10]}"
            )
        else:
            print(f"[CrackDataset] Found {len(self.pairs)} pairs "
                  f"in {self.images_dir.name} / {self.masks_dir.name}")

        self.tf_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        self.tf_mask = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return self.tf_img(img), self.tf_mask(mask)
