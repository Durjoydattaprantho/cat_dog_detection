import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import build_model
from dataset import IMAGENET_MEAN, IMAGENET_STD

class CatDogPredictor:
    def __init__(self, checkpoint_path="output/best_cat_dog_model.pth", device=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint '{checkpoint_path}' not found. Please train the model first.")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes = checkpoint.get("classes", ["cat", "dog"])
        self.backbone = checkpoint.get("backbone", "mobilenet_v2")

        # Build and load model
        self.model = build_model(backbone=self.backbone, num_classes=len(self.classes), pretrained=False, freeze_base=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def predict(self, image_input):
        """
        Accepts either a file path (str), a PIL Image, or bytes.
        Returns prediction details dict.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            # Assume file-like or bytes
            image = Image.open(image_input).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)

        conf, pred_idx = torch.max(probs, dim=0)
        predicted_class = self.classes[pred_idx.item()]
        confidence = conf.item()

        all_probs = {self.classes[i]: round(probs[i].item(), 4) for i in range(len(self.classes))}

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": all_probs,
            "is_cat": predicted_class.lower() == "cat",
            "is_dog": predicted_class.lower() == "dog"
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat vs Dog Image Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="output/best_cat_dog_model.pth", help="Model checkpoint path")
    args = parser.parse_args()

    predictor = CatDogPredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.image)
    print(f"\nResult: {result['predicted_class'].upper()} (Confidence: {result['confidence']*100:.2f}%)")
    print(f"Probabilities: {result['probabilities']}")
