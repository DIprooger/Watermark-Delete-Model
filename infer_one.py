import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

# ---------- 1. загрузка модели ----------
ckpt = torch.load("watermark_model.pth", map_location="cpu")
ckpt = {k.replace("model.", "", 1): v for k, v in ckpt.items()}

model = smp.create_model(
    arch="UnetPlusPlus",
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    encoder_weights=None
)
model.load_state_dict(ckpt, strict=False)
model.eval()

# ---------- 2. директории ----------
input_dir = "image_for_cleare"
output_dir = "output_cleaned"
os.makedirs(output_dir, exist_ok=True)

TARGET = (512, 512)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------- 3. обработка всех изображений ----------
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        continue

    path = os.path.join(input_dir, filename)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"⚠️ Не удалось прочитать файл: {filename}")
        continue

    img_res = cv2.resize(img_bgr, TARGET)
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    img_rgb = (img_rgb - mean) / std

    input_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)

    mask_prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    H, W = img_bgr.shape[:2]
    mask_prob = cv2.resize(mask_prob, (W, H), interpolation=cv2.INTER_LINEAR)

    mask_bin = (mask_prob > 0.35).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    inpainted = cv2.inpaint(img_bgr, mask_bin, 5, cv2.INPAINT_TELEA)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, inpainted)
    print(f"✅ {filename} очищено → {out_path}")

print("🎉 Все изображения обработаны.")

