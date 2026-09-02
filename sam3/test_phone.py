import numpy as np
import torch

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ===== 1. 修改成你自己的图片路径 =====
IMAGE_PATH = r"C:\Users\Admin\Desktop\test_phone.jpg"

# ===== 2. 文本目标 =====
PROMPT = "phone"


print("Loading SAM3...")

model = build_sam3_image_model()
processor = Sam3Processor(model)

print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")


print(f"Running prompt: {PROMPT}")

with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        state = processor.set_image(image)

        output = processor.set_text_prompt(
            state=state,
            prompt=PROMPT
        )


masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

print("\n========== RESULT ==========")
print("Instances:", len(masks))
print("Boxes:", boxes)
print("Scores:", scores)


# ===== 3. 保存每个mask =====

for i, mask in enumerate(masks):

    # Tensor -> numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # 去掉多余维度，例如 [1,H,W] -> [H,W]
    mask = np.squeeze(mask)

    # 变成0/255二值mask
    mask_img = (mask > 0.5).astype(np.uint8) * 255

    Image.fromarray(mask_img).save(
        f"sam3_mask_{i}.png"
    )

    print(f"Saved: sam3_mask_{i}.png")


# ===== 4. 生成overlay =====

original = np.array(image).astype(np.float32)

overlay = original.copy()

for mask in masks:

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.squeeze(mask) > 0.5

    # 为了方便观察，把mask区域提亮
    overlay[mask] = (
        overlay[mask] * 0.5
        + np.array([255, 0, 0]) * 0.5
    )

overlay = np.clip(overlay, 0, 255).astype(np.uint8)

Image.fromarray(overlay).save(
    "sam3_overlay.jpg"
)

print("\nSaved: sam3_overlay.jpg")
print("Done!")
