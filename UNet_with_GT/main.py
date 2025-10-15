import os, glob, csv
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from unet_model import UNet  # or from unet import UNet

import os, csv

def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def csv_append_row(path: str, row_dict: dict):
    _ensure_parent_dir(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


# -------------------- SPEED / GPU SETTINGS --------------------
torch.backends.cudnn.benchmark = True                   # autotune convs for fixed input size
torch.set_float32_matmul_precision("high")              # PyTorch 2.x matmul speed
USE_AMP = True                                          # mixed precision
CHANNELS_LAST = True

# -------------------- YOUR PATHS --------------------
TRAIN_POS_DIR = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\train\Positive"
TRAIN_NEG_DIR = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\train\Negative"
TRAIN_GT_DIR  = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\bw\train"

VAL_POS_DIR   = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\val\images\Positive"
VAL_NEG_DIR   = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\val\images\Negative"
VAL_GT_DIR    = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\bw\val"   # masks for *positive* val images

MODEL_PATH    = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\models\unet_model_bruno2.pth"
CURVE_PNG     = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\models\loss_curves_bruno2.png"
CSV_PATH      = r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\models\training_log_bruno2.csv"

EXTS = (".jpg", ".JPG")  # tolerate case variants

# -------------------- DATASET --------------------
class PatchDataset(Dataset):
    """
    pairs: list of (img_path, mask_path or None). If mask is None -> create all-zero mask.
    """
    def __init__(self, pairs, img_size=224):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, p):
        img = Image.open(p).convert("RGB")
        if self.img_size:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC->CHW
        return torch.from_numpy(arr)

    def _load_mask_or_zero(self, p, H, W):
        if p is None:
            return torch.zeros((1, H, W), dtype=torch.float32)  # negative -> all-zero mask
        m = Image.open(p).convert("L")
        if self.img_size:
            m = m.resize((self.img_size, self.img_size), Image.NEAREST)
        m = (np.asarray(m, dtype=np.uint8) > 0).astype(np.float32)
        m = m[None, ...]  # (1,H,W)
        return torch.from_numpy(m)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        x = self._load_image(img_path)
        _, H, W = x.shape
        y = self._load_mask_or_zero(mask_path, H, W)
        return x, y

# -------------------- HELPERS --------------------
def _glob_all(folder):
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return files

def _find_mask_for(stem: str, gt_dir: str):
    """
    Prefer 'bw_{stem}.jpg/JPG'; fallback to '{stem}.jpg/JPG' if some masks lack 'bw_'.
    Returns mask path or None.
    """
    candidates = []
    for ext in EXTS:
        candidates.append(os.path.join(gt_dir, f"bw_{stem}{ext}"))
    for ext in EXTS:
        candidates.append(os.path.join(gt_dir, f"{stem}{ext}"))  # fallback
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# -------------------- BUILD SPLITS --------------------
def build_train_pairs():
    """
    Train = Positives with masks from TRAIN_GT_DIR (prefer bw_{stem}.jpg)
          + Negatives with mask=None (all-zero).
    """
    pairs = []

    # positives + masks
    for ip in _glob_all(TRAIN_POS_DIR):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = _find_mask_for(stem, TRAIN_GT_DIR)
        if mp is None:
            print(f"[WARN] Missing TRAIN mask for: {ip}")
            continue
        pairs.append((ip, mp))

    # negatives -> zero masks
    if os.path.isdir(TRAIN_NEG_DIR):
        for ip in _glob_all(TRAIN_NEG_DIR):
            pairs.append((ip, None))

    if not pairs:
        raise RuntimeError("No training samples found. Check TRAIN paths and names.")
    return pairs

def build_val_pairs():
    """
    Val = Positives with masks from VAL_GT_DIR (prefer bw_{stem}.jpg)
        + Negatives with mask=None (all-zero).
    """
    pairs = []

    # positives + masks
    for ip in _glob_all(VAL_POS_DIR):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = _find_mask_for(stem, VAL_GT_DIR)
        if mp is None:
            print(f"[WARN] Missing VAL mask for positive: {ip}")
            continue
        pairs.append((ip, mp))

    # negatives -> zero masks
    if os.path.isdir(VAL_NEG_DIR):
        for ip in _glob_all(VAL_NEG_DIR):
            pairs.append((ip, None))

    if not pairs:
        raise RuntimeError("No validation samples found. Check VAL paths and names.")
    return pairs

# -------------------- TRAINING --------------------
def main():
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 12          # 4070 Laptop GPU: try 8, then 12–16 if it fits
    EPOCHS = 10
    IMG_SIZE = 224

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] device={torch.cuda.get_device_name(0)}")

    train_pairs = build_train_pairs()
    val_pairs   = build_val_pairs()

    n_train_pos = sum(1 for _, m in train_pairs if m is not None)
    n_train_neg = sum(1 for _, m in train_pairs if m is None)
    n_val_pos   = sum(1 for _, m in val_pairs   if m is not None)
    n_val_neg   = sum(1 for _, m in val_pairs   if m is None)

    print(f"[INFO] Train: {len(train_pairs)} (pos={n_train_pos}, neg={n_train_neg})")
    print(f"[INFO] Val:   {len(val_pairs)} (pos={n_val_pos}, neg={n_val_neg})")

    train_ds = PatchDataset(train_pairs, img_size=IMG_SIZE)
    val_ds   = PatchDataset(val_pairs,   img_size=IMG_SIZE)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # probe
    xb, yb = next(iter(train_loader))
    print(f"[DEBUG] first train batch shapes: x={tuple(xb.shape)} y={tuple(yb.shape)}")

    model = UNet(in_channels=3, num_classes=1)
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # --- DISABLE torch.compile (it was causing your crash) ---
    # If you want to try later: wrap in try/except and only enable if stable.
    # try:
    #     model = torch.compile(model)
    #     print("[INFO] torch.compile enabled")
    # except Exception:
    #     print("[INFO] torch.compile failed, running eager mode")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # New AMP API (fixes FutureWarning)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and USE_AMP))

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        # --- train ---
        model.train()
        tr_sum, tr_batches = 0.0, 0
        for img, mask in tqdm(train_loader, desc="Train", leave=False):
            if CHANNELS_LAST:
                img = img.to(memory_format=torch.channels_last)
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and USE_AMP)):
                pred = model(img)
                loss = criterion(pred, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_sum += loss.item()
            tr_batches += 1
        train_loss = tr_sum / max(tr_batches, 1)

        # --- validate ---
        model.eval()
        vl_sum, vl_batches = 0.0, 0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc="Val", leave=False):
                if CHANNELS_LAST:
                    img = img.to(memory_format=torch.channels_last)
                img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda" and USE_AMP)):
                    pred = model(img)
                    loss = criterion(pred, mask)
                vl_sum += loss.item()
                vl_batches += 1
        val_loss = vl_sum / max(vl_batches, 1)

        csv_append_row(CSV_PATH, {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        })

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print("-" * 50)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        print("-" * 50)

    # --- save model ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    # --- plot & save curves ---
    try:
        plt.figure()
        plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
        plt.plot(history["epoch"], history["val_loss"],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend(); plt.tight_layout()
        os.makedirs(os.path.dirname(CURVE_PNG), exist_ok=True)
        plt.savefig(CURVE_PNG, dpi=150)
        print(f"[INFO] Loss curves saved to {CURVE_PNG}")
        plt.show()
    except Exception as e:
        print(f"[WARN] Could not plot/save curves: {e}")

    # --- save CSV log ---
    try:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])
            for ep, tr, vl in zip(history["epoch"], history["train_loss"], history["val_loss"]):
                w.writerow([ep, tr, vl])
        print(f"[INFO] Training log saved to {CSV_PATH}")
    except Exception as e:
        print(f"[WARN] Could not save CSV log: {e}")

if __name__ == "__main__":
    main()
