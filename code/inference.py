# src/inference.py

import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Models import DeepLabV3PlusEnhanced


class InferenceEngine:
    def __init__(self, model_path, num_classes=5, encoder="resnet50", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepLabV3PlusEnhanced(
            encoder=encoder,
            num_classes=num_classes,
            encoder_weights="imagenet"
        ).to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"Model loaded from: {model_path}")

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

    def infer_image(self, image_path):
        # Preprocess the input image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Perform inference
            seg_out, price_out = self.model(input_tensor)
            prob = torch.softmax(seg_out, dim=1)
            max_conf, pred_class = torch.max(prob, dim=1)

        pred_mask = pred_class.squeeze().cpu().numpy().astype(np.uint8)
        confidence = max_conf.squeeze().cpu().numpy()
        price = price_out.item()

        return pred_mask, confidence, price

    def run_batch_inference(self, input_dir, output_dir):
        # Run inference for all supported images in the input directory
        os.makedirs(output_dir, exist_ok=True)
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        for fname in os.listdir(input_dir):
            if not fname.lower().endswith(supported_ext):
                continue

            image_path = os.path.join(input_dir, fname)
            base = os.path.splitext(fname)[0]

            mask_path = os.path.join(output_dir, f"{base}_mask.png")
            conf_path = os.path.join(output_dir, f"{base}_conf.npy")

            pred_mask, confidence, pred_price = self.infer_image(image_path)
            Image.fromarray(pred_mask).save(mask_path)
            np.save(conf_path, confidence)

            print(f"{fname} - Estimated repair cost: {pred_price:.2f}")
            print(f"Saved mask: {mask_path}")
            print(f"Saved confidence map: {conf_path}")


if __name__ == "__main__":
    test_input_dir = "/home/lhh5785/car/data/code/test_input"
    result_output_dir = "/home/lhh5785/car/data/code/result_img"
    weight_path = "/home/lhh5785/car/data/weight/DeepLabV3_damage_best_epoch19_mIoU43.7.pt"

    engine = InferenceEngine(model_path=weight_path)
    engine.run_batch_inference(test_input_dir, result_output_dir)
