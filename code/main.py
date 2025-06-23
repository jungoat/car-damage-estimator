# main.py

import os
import sys
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lovasz import lovasz_softmax
from src.Train import Trainer
from src.Evaluation import Evaluation
from src.Models import DeepLabV3PlusEnhanced
from segmentation_models_pytorch.losses import FocalLoss as SMPFocalLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_total_loss():
    focal = SMPFocalLoss(mode='multiclass', gamma=2.0)
    def total_loss(seg_pred, seg_target, price_pred, price_target, lambda_price=1e-6):
        seg_loss = 0.4 * lovasz_softmax(seg_pred, seg_target, per_image=False) + 0.6 * focal(seg_pred, seg_target)
        price_loss = F.l1_loss(price_pred, price_target)
        return seg_loss + lambda_price * price_loss, seg_loss, price_loss
    return total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--resume_path', type=str, default=None)
    args = parser.parse_args()

    set_seed(1230)
    print(f'GPU device index: {torch.cuda.current_device()}')

    model = DeepLabV3PlusEnhanced(
        encoder="resnet50",
        num_classes=5,
        encoder_weights="imagenet"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()}개의 GPU를 사용합니다.")
        model = torch.nn.DataParallel(model)

    criterion = get_total_loss()

    if args.resume_path:
        print(f"🔁 모델 weight 로드 중: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise ValueError("지원되지 않는 checkpoint 형식입니다.")
        print("모델 weight 로드 완료")

    if args.dataset != 'test':
        print("학습 시작")
        trainer = Trainer(
            ails=args.task,
            train_dir="/home/lhh5785/car/data/datainfo/damage_train.json",
            val_dir="/home/lhh5785/car/data/datainfo/damage_val.json",
            img_base_path="/home/lhh5785/car/data/Dataset/1.원천데이터/part/damage",
            size=512,
            model=model,
            label=args.label,
            n_class=5,
            optimizer=torch.optim.Adam,
            criterion=criterion,
            epochs=100,
            batch_size=32,
            encoder_lr=1e-6,
            decoder_lr=3e-4,
            weight_decay=0,
            device="cuda",
            start_epoch=0
        )
        trainer.train()
    else:
        print("테스트 모드: 학습을 건너뜁니다.")

    print("평가 시작")
    set_seed(12)
    evaluation = Evaluation(
        eval_dir=f"../data/datainfo/damage_{args.dataset}.json",
        size=512,
        model=model,
        weight_paths=[],
        device='cuda',
        batch_size=32,
        ails=f"../data/result_log/[{args.task}]_{args.dataset}_evaluation_log.json",
        criterion=torch.nn.CrossEntropyLoss(),
        img_base_path="../data/Dataset/1.원천데이터/damage"
    )
    evaluation.evaluation()
