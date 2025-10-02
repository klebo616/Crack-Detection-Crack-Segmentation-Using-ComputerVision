import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model_unet import UNet            # or from unet import UNet
from crack_dataset import CrackDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8                     # try smaller to ensure at least 1 batch
    EPOCHS = 10

    images_dir = r"C:/Users/bob1k/Desktop/MASTER/Thesis/CNN/images_selective"       # <-- update to your current paths
    masks_dir  = r"C:/Users/bob1k/Desktop/MASTER/Thesis/CNN/masks_OpenCV"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CrackDataset(images_dir, masks_dir, img_size=512)

    # ✅ CORRECT split using integer lengths (not fractions)
    n_total = len(ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    print(f"[INFO] total={n_total}  train={n_train}  val={n_val}")

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(ds, [n_train, n_val], generator=generator)

    print(f"[INFO] len(train_dataset)={len(train_dataset)}  len(val_dataset)={len(val_dataset)}")

    # DataLoaders (keep workers=0 on Windows to avoid multiprocessing glitches)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 🔎 PROBE: try to pull 1 batch so we see errors early
    try:
        xb, yb = next(iter(train_dataloader))
        print(f"[DEBUG] first batch shapes: x={tuple(xb.shape)} y={tuple(yb.shape)}")
    except StopIteration:
        print("[ERROR] Train dataloader is empty. Check your split lengths and batch size.")
        raise
    except Exception as e:
        print("[ERROR] Exception while loading first batch:")
        raise

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        model.train()
        train_running_loss = 0.0
        batches = 0

        for img, mask in tqdm(train_dataloader, desc="Train", leave=False):
            img  = img.float().to(device)
            mask = mask.float().to(device)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            batches += 1

        if batches == 0:
            print("[ERROR] No training batches iterated. Dataloader empty.")
            break

        train_loss = train_running_loss / batches

        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for img, mask in tqdm(val_dataloader, desc="Val", leave=False):
                img  = img.float().to(device)
                mask = mask.float().to(device)
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
                val_batches += 1

        val_loss = val_running_loss / max(val_batches, 1)
        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

  # After all epochs finish
    MODEL_PATH = r"C:\Users\bob1k\Desktop\MASTER\Thesis\CNN\models\unetModel1"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")
