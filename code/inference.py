import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# 상위 폴더(src) 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Models import Unet


class InferenceEngine:
    def __init__(self, model_path, num_classes=5, encoder="resnet50", device=None):
        # GPU/CPU 장치 설정
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 초기화 (여기선 Unet 사용)
        self.model = Unet(
            encoder=encoder,
            num_classes=num_classes,
            encoder_weights="imagenet"
        ).to(self.device)

        # 멀티 GPU 지원
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 가중치 불러오기
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()
        print(f"Model loaded from: {model_path}")

        # 입력 이미지 전처리 파이프라인
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

    def infer_image(self, image_path):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 추론 수행
            seg_out, price_out = self.model(input_tensor)  # 모델이 두 개 출력한다고 가정
            prob = torch.softmax(seg_out, dim=1)
            max_conf, pred_class = torch.max(prob, dim=1)

        # 예측 mask와 confidence map
        pred_mask = pred_class.squeeze().cpu().numpy().astype(np.uint8)
        confidence = max_conf.squeeze().cpu().numpy()
        price = price_out.item()

        return pred_mask, confidence, price

    def run_batch_inference(self, input_dir, output_dir):
        # 폴더 내 모든 이미지에 대해 추론 실행
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
