import os, glob, csv
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
import torchvision.transforms as T
from unet_model import UNet

# -------------------- CONFIG --------------------
CFG = dict(
    lr=3e-4, wd=1e-4, batch_size=16, epochs=60, img_size=224,
    use_amp=True, channels_last=True, num_workers=4,

    # file paths
    model_path=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\models\unet_model_bruno10.pth",
    curve_png=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\models\loss_curves_bruno10.png",
    csv_path=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\models\training_log_bruno10.csv",

    # data paths
    train_pos=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\train\Positive",
    train_neg=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\train\Negative",
    train_gt=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\bw\train",
    val_pos=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\val\images\Positive",
    val_neg=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\patches\val\images\Negative",
    val_gt=r"C:\Users\bob1k\Desktop\MASTER\Thesis\UNet\dataset_bruno\bw\val",
    exts=(".jpg", ".JPG"),

    # training control
    early_patience=1000,
    scheduler_patience=1,
    scheduler_factor=0.5,

    # extra checks
    check_aug_diff=True,

    # loss options
    pos_weight_value=1.0,
    use_dice_loss=True,
    dice_weight=1.0,
)

# -------------------- UTILS --------------------
def ensure_dir(path): os.makedirs(os.path.dirname(path), exist_ok=True)


def csv_append_row(path, row_dict):
    ensure_dir(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


def glob_all(folder, exts):
    out = []
    for e in exts:
        out += glob.glob(os.path.join(folder, f"*{e}"))
    return out


def find_mask_for(stem, gt_dir, exts):
    cands = [os.path.join(gt_dir, f"bw_{stem}{e}") for e in exts] + \
            [os.path.join(gt_dir, f"{stem}{e}") for e in exts]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


# -------------------- DATA --------------------
class PatchDataset(Dataset):
    def __init__(self, pairs, img_size=224):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        if self.img_size:
            img = img.resize((self.img_size, self.img_size))
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(x.transpose(2, 0, 1))
        if mp is None:
            _, H, W = x.shape
            y = torch.zeros((1, H, W), dtype=torch.float32)
        else:
            m = Image.open(mp).convert("L")
            if self.img_size:
                m = m.resize((self.img_size, self.img_size))
            m = (np.asarray(m, dtype=np.uint8) > 0).astype(np.float32)[None, ...]
            y = torch.from_numpy(m)
        return x, y


class AlbumentationsWrapper(Dataset):
    def __init__(self, base_ds, geo_tfm):
        self.base = base_ds
        self.geo_tfm = geo_tfm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        img = x.numpy().transpose(1, 2, 0)
        msk = y.numpy().transpose(1, 2, 0)
        out = self.geo_tfm(image=img, mask=msk)
        xi = torch.from_numpy(out["image"].transpose(2, 0, 1)).float()
        yi = torch.from_numpy(out["mask"].transpose(2, 0, 1)).float().clamp(0, 1)
        return xi, yi


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.04):
        super().__init__()
        self.std = std

    def forward(self, x):
        return (x + torch.randn_like(x) * self.std).clamp(0, 1)


class AugmentWrapper(Dataset):
    def __init__(self, base_ds, tfm):
        self.base = base_ds
        self.tfm = tfm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return self.tfm(x), y


def build_pairs(pos_dir, neg_dir, gt_dir, exts):
    pairs = []
    for ip in glob_all(pos_dir, exts):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = find_mask_for(stem, gt_dir, exts)
        if mp:
            pairs.append((ip, mp))
        else:
            print(f"[WARN] Missing mask for: {ip}")
    if os.path.isdir(neg_dir):
        for ip in glob_all(neg_dir, exts):
            pairs.append((ip, None))
    return pairs


# -------------------- AUG BUILDER --------------------
def build_geo_tfm():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(CFG["img_size"], CFG["img_size"],
                            scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10,
                           rotate_limit=10, border_mode=0, p=0.6),
    ], additional_targets={'mask': 'mask'})


def build_img_tfm():
    return T.Compose([
        T.ColorJitter(brightness=0.42, contrast=0.4),
        T.RandomApply([T.GaussianBlur(3, (0.35, 1.0))], p=0.8),
        T.RandomApply([AddGaussianNoise(std=0.05)], p=0.9),
    ])


# -------------------- METRICS --------------------
@torch.no_grad()
def dice_iou_from_logits(logits, target, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * target).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    targ_sum = target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (pred_sum + targ_sum + eps)
    iou = (inter + eps) / (pred_sum + targ_sum - inter + eps)
    return dice.mean().item(), iou.mean().item()


def dice_loss_from_logits(logits, target, eps=1e-7):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    pred_sum = probs.sum(dim=(2, 3))
    targ_sum = target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (pred_sum + targ_sum + eps)
    return 1.0 - dice.mean()


# -------------------- TRAIN/VAL --------------------
def train_one_epoch(model, loader, device, scaler, bce, optimizer, channels_last):
    model.train()
    total = 0.0
    n = 0
    for img, mask in tqdm(loader, desc="Train", leave=False):
        if channels_last:
            img = img.to(memory_format=torch.channels_last)
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(img)
            loss_bce = bce(logits, mask)
            if CFG["use_dice_loss"]:
                loss_dice = dice_loss_from_logits(logits, mask)
                loss = loss_bce + CFG["dice_weight"] * loss_dice
            else:
                loss = loss_bce
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, scaler, bce, channels_last):
    model.eval()
    total = 0.0
    n = 0
    dice_list = []
    iou_list = []
    for img, mask in tqdm(loader, desc="Val", leave=False):
        if channels_last:
            img = img.to(memory_format=torch.channels_last)
        img, mask = img.to(device), mask.to(device)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(img)
            loss_bce = bce(logits, mask)
            if CFG["use_dice_loss"]:
                loss = loss_bce + CFG["dice_weight"] * dice_loss_from_logits(logits, mask)
            else:
                loss = loss_bce
        total += loss.item()
        n += 1
        d, i = dice_iou_from_logits(logits, mask)
        dice_list.append(d)
        iou_list.append(i)
    return total / max(n, 1), float(np.mean(dice_list)), float(np.mean(iou_list))


def _wif(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# -------------------- MAIN FIT --------------------
def fit():
    # -------- Determinism & seeds --------
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # full reproducibility

    set_seed(42)
    torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] cuda={torch.cuda.is_available()} | device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # ---- Aug diff diagnostic (optional) ----
    if CFG["check_aug_diff"]:
        base = PatchDataset([(os.path.join(CFG["train_pos"], os.listdir(CFG["train_pos"])[0]), None)], CFG["img_size"])[0][0]
        tfm = build_img_tfm()
        diffs = [(tfm(base.clone()) - base).abs().mean().item() for _ in range(100)]
        thr = 0.08
        aff = sum(d > thr for d in diffs) / len(diffs)
        print(f"[AUG CHECK] avg diff={np.mean(diffs):.3f} | std={np.std(diffs):.3f} | affected(>{thr})={aff * 100:.1f}%")

    # -------- Build file pairs --------
    train_pairs = build_pairs(CFG["train_pos"], CFG["train_neg"], CFG["train_gt"], CFG["exts"])
    val_pairs = build_pairs(CFG["val_pos"], CFG["val_neg"], CFG["val_gt"], CFG["exts"])
    print(f"[INFO] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # -------- Datasets --------
    base_train = PatchDataset(train_pairs, CFG["img_size"])
    base_val = PatchDataset(val_pairs, CFG["img_size"])
    train_ds = AlbumentationsWrapper(base_train, build_geo_tfm())
    train_ds = AugmentWrapper(train_ds, build_img_tfm())
    val_ds = base_val

    # -------- DataLoaders --------
    kw = dict(
        num_workers=CFG["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_wif
    )
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, **kw)

    # -------- Model / Loss / Opt --------
    model = UNet(in_channels=3, num_classes=1)
    if CFG["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    pw = 1.0 if CFG["pos_weight_value"] is None else float(CFG["pos_weight_value"])
    pos_w = torch.tensor([pw], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=CFG["scheduler_factor"], patience=CFG["scheduler_patience"], verbose=True
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and CFG["use_amp"]))

    # -------- Train loop --------
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "dice": [], "iou": []}
    best_val = float("inf")
    bad = 0
    best_path = CFG["model_path"].replace(".pth", "_best.pth")

    for ep in tqdm(range(1, CFG["epochs"] + 1), desc="Epochs"):
        tr = train_one_epoch(model, train_loader, device, scaler, bce, optimizer, CFG["channels_last"])
        vl, dsc, iu = evaluate(model, val_loader, device, scaler, bce, CFG["channels_last"])
        scheduler.step(vl)

        hist["epoch"].append(ep)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(vl)
        hist["dice"].append(dsc)
        hist["iou"].append(iu)
        csv_append_row(CFG["csv_path"], {"epoch": ep, "train_loss": tr, "val_loss": vl, "dice": dsc, "iou": iu})
        print(f"Epoch {ep:03d} | train={tr:.4f} | val={vl:.4f} | dice={dsc:.3f} | IoU={iu:.3f}")

        if vl < best_val - 1e-4:
            best_val = vl
            bad = 0
            ensure_dir(best_path)
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] saved best -> {best_path}")
        else:
            bad += 1
            if bad >= CFG["early_patience"]:
                print(f"[INFO] early stop at epoch {ep}")
                break

    ensure_dir(CFG["model_path"])
    torch.save(model.state_dict(), CFG["model_path"])
    print(f"[INFO] saved last -> {CFG['model_path']}")

    # -------- Plot curves --------
    try:
        ensure_dir(CFG["curve_png"])
        fig, ax1 = plt.subplots()
        ax1.plot(hist["epoch"], hist["train_loss"], label="Train Loss")
        ax1.plot(hist["epoch"], hist["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle=":")
        ax2 = ax1.twinx()
        ax2.plot(hist["epoch"], hist["dice"], linestyle="--", color="green", label="Dice")
        ax2.plot(hist["epoch"], hist["iou"], linestyle="--", color="orange", label="IoU")
        ax2.set_ylabel("Score")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        plt.tight_layout()
        plt.savefig(CFG["curve_png"], dpi=150)
        plt.close()
        print(f"[INFO] curves -> {CFG['curve_png']}")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")


if __name__ == "__main__":
    fit()
