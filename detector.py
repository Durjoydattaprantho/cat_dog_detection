import os
import io
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from model import build_model
from dataset import IMAGENET_MEAN, IMAGENET_STD

class CatDogDetector:
    """
    Advanced Cat and Dog Detection System:
    Combines Deep Classification with Grad-CAM Object Localization to detect:
    1. Category (Cat vs Dog)
    2. Confidence Score & Class Probabilities
    3. Spatial Object Bounding Box(es) enclosing the animal
    4. Attention Heatmap (Grad-CAM Saliency)
    """
    def __init__(self, checkpoint_path="output/best_cat_dog_model.pth", device=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found. Please train first.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint.get("classes", ["cat", "dog"])
        self.backbone = checkpoint.get("backbone", "mobilenet_v2")

        self.model = build_model(backbone=self.backbone, num_classes=len(self.classes), pretrained=False, freeze_base=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def detect(self, image_input, box_threshold=0.35):
        """
        Runs full detection pipeline on image.
        Returns:
            dict containing prediction, confidence, bounding boxes,
            and base64-encoded annotated detection image and heatmap image.
        """
        if isinstance(image_input, (str, os.PathLike)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            # Assume file-like object
            pil_img = Image.open(image_input).convert("RGB")

        orig_w, orig_h = pil_img.size
        orig_np = np.array(pil_img)
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Hook into features layer for Grad-CAM
        feature_maps = None
        def hook_fn(module, inp, out):
            nonlocal feature_maps
            feature_maps = out

        hook_handle = self.model.features.register_forward_hook(hook_fn)

        outputs = self.model(tensor)
        hook_handle.remove()

        probs = F.softmax(outputs, dim=1).squeeze(0)
        conf_val, pred_idx = torch.max(probs, dim=0)
        pred_class = self.classes[pred_idx.item()]
        confidence = conf_val.item()

        # Compute Grad-CAM
        self.model.zero_grad()
        score = outputs[0, pred_idx]
        feat_grad = torch.autograd.grad(score, feature_maps)[0]
        weights = feat_grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1).squeeze().relu().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = cv2.resize(cam, (orig_w, orig_h))

        # Generate Heatmap
        heatmap = (cam_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(orig_bgr, 0.65, heatmap_color, 0.35, 0)

        # Extract Bounding Boxes
        mask = (cam_resized > box_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (orig_w * orig_h) * 0.03
        detected_boxes = []

        annotated_bgr = orig_bgr.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Expand box slightly for natural look
                pad_x = int(w * 0.05)
                pad_y = int(h * 0.05)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(orig_w, x + w + pad_x)
                y2 = min(orig_h, y + h + pad_y)

                detected_boxes.append({
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "area": int(area)
                })

        # If no contour passed area threshold, use largest activation region
        if not detected_boxes:
            y_indices, x_indices = np.where(cam_resized > 0.25)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x1, x2 = int(np.min(x_indices)), int(np.max(x_indices))
                y1, y2 = int(np.min(y_indices)), int(np.max(y_indices))
                detected_boxes.append({
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                })
            else:
                detected_boxes.append({
                    "x": int(orig_w * 0.1),
                    "y": int(orig_h * 0.1),
                    "width": int(orig_w * 0.8),
                    "height": int(orig_h * 0.8),
                    "area": int(orig_w * orig_h * 0.64)
                })

        # Sort by area descending (primary detection first)
        detected_boxes.sort(key=lambda b: b["area"], reverse=True)

        # Draw Bounding Box and Stylish HUD Labels on annotated_bgr
        color = (34, 197, 94) if pred_class == "cat" else (59, 130, 246)  # Green for Cat, Blue for Dog
        emoji = "🐱" if pred_class == "cat" else "🐶"

        for i, box in enumerate(detected_boxes):
            bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]

            # Main rectangle
            cv2.rectangle(annotated_bgr, (bx, by), (bx + bw, by + bh), color, 3)

            # Modern corner brackets
            corner_len = min(25, bw // 4, bh // 4)
            corner_thick = 5
            # Top-left
            cv2.line(annotated_bgr, (bx, by), (bx + corner_len, by), color, corner_thick)
            cv2.line(annotated_bgr, (bx, by), (bx, by + corner_len), color, corner_thick)
            # Top-right
            cv2.line(annotated_bgr, (bx + bw, by), (bx + bw - corner_len, by), color, corner_thick)
            cv2.line(annotated_bgr, (bx + bw, by), (bx + bw, by + corner_len), color, corner_thick)
            # Bottom-left
            cv2.line(annotated_bgr, (bx, by + bh), (bx + corner_len, by + bh), color, corner_thick)
            cv2.line(annotated_bgr, (bx, by + bh), (bx, by + bh - corner_len), color, corner_thick)
            # Bottom-right
            cv2.line(annotated_bgr, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color, corner_thick)
            cv2.line(annotated_bgr, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color, corner_thick)

            # Tag label banner
            if i == 0:
                label_text = f"{pred_class.upper()} {confidence*100:.1f}%"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = max(0.55, min(1.0, orig_w / 600.0))
                (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, 2)

                tag_y1 = max(0, by - txt_h - 14)
                tag_y2 = by
                cv2.rectangle(annotated_bgr, (bx, tag_y1), (bx + txt_w + 16, tag_y2), color, -1)
                cv2.putText(annotated_bgr, label_text, (bx + 8, by - 6), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert images to base64
        def to_b64(bgr_mat):
            _, buf = cv2.imencode('.jpg', bgr_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            return base64.b64encode(buf).decode('utf-8')

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "cat": round(probs[0].item(), 4),
                "dog": round(probs[1].item(), 4)
            },
            "bounding_boxes": detected_boxes,
            "annotated_image_b64": f"data:image/jpeg;base64,{to_b64(annotated_bgr)}",
            "heatmap_image_b64": f"data:image/jpeg;base64,{to_b64(heatmap_overlay)}",
            "original_image_b64": f"data:image/jpeg;base64,{to_b64(orig_bgr)}",
            "image_size": {"width": orig_w, "height": orig_h}
        }
