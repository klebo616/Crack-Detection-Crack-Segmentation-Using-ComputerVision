import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

from model_unet import UNet   # or model_unet if that's your file name

# ---------------------- CONFIG ----------------------
MODEL_PATH   = Path(r"C:\Users\bob1k\Desktop\MASTER\Thesis\CNN\models\unetModel1")
IMAGES_DIR   = Path(r"C:\Users\bob1k\Desktop\MASTER\Thesis\CNN\Prediction_images")
OUT_DIR      = Path(r"C:\Users\bob1k\Desktop\MASTER\Thesis\CNN\preds")

IMG_SIZE     = 512      # must match training resize
THRESH       = 0.5      # probability threshold
SHOW_RESULTS = True     # set False if you only want to save
# ----------------------------------------------------


def _to_numpy_img(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    return x.clamp(0, 1).numpy()


def _to_numpy_mask(t: torch.Tensor) -> np.ndarray:
    m = t.detach().cpu()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    return m.numpy()


@torch.no_grad()
def batch_inference(images_dir: Path, model_pth: Path, out_dir: Path, device: str):
    assert images_dir.exists(), f"Image folder not found: {images_dir}"
    assert model_pth.exists(),  f"Model file not found: {model_pth}"

    # model
    model = UNet(in_channels=3, num_classes=1).to(device)
    state = torch.load(model_pth, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state["model_state"])
    model.eval()

    # transforms
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    out_dir.mkdir(parents=True, exist_ok=True)

    # iterate all images in folder
    img_files = [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    if not img_files:
        raise RuntimeError(f"No images found in {images_dir}")

    for img_path in img_files:
        print(f"[INFO] Processing {img_path.name}")

        # load + transform
        img_pil = Image.open(str(img_path)).convert("RGB")
        img_t   = tf(img_pil)
        x       = img_t.unsqueeze(0).to(device)

        # forward
        logits = model(x)
        probs  = torch.sigmoid(logits)
        pred   = (probs > THRESH).float()
        pred_np = _to_numpy_mask(pred[0])

        # save mask
        save_path = out_dir / f"{img_path.stem}_pred.png"
        Image.fromarray((pred_np * 255).astype(np.uint8)).save(str(save_path))

        if SHOW_RESULTS:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(_to_numpy_img(img_t)); axes[0].set_title("Image"); axes[0].axis("off")
            axes[1].imshow(pred_np, cmap="gray");  axes[1].set_title("Pred Mask"); axes[1].axis("off")
            plt.suptitle(img_path.name)
            plt.tight_layout(); plt.show()

        print(f"[OK] Saved prediction to {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    batch_inference(IMAGES_DIR, MODEL_PATH, OUT_DIR, device)
