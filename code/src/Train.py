import os
import json
import datetime
from pytz import timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

import Models as models
from Datasets import Datasets
from Utils import label_accuracy_score, add_hist


class Trainer:
    def __init__(self,
                 train_dir,
                 val_dir,
                 size,
                 label,
                 model,
                 n_class,
                 criterion,
                 optimizer,
                 device,
                 epochs,
                 batch_size,
                 encoder_lr,
                 decoder_lr,
                 weight_decay,
                 ails,
                 train_img_base_path=None,
                 val_img_base_path=None,
                 transform=None,
                 lr_scheduler=None,
                 start_epoch=None):

        # Unet 모델 내부 구조(model.model 사용)
        self.model = model.model
        self.n_class = n_class
        self.criterion = criterion
        self.device = device

        # 학습 관련 기본 설정
        self.epochs = epochs
        self.batch_size = batch_size
        self.label = label
        self.one_channel = (label is not None)  # label 있으면 one-channel segmentation

        # 데이터셋 로드
        self.train_dataset = Datasets(train_dir, 'train', size, label, self.one_channel, train_img_base_path, transform)
        self.val_dataset = Datasets(val_dir, 'val', size, label, self.one_channel, val_img_base_path)

        # Optimizer 설정 (encoder/decoder 서로 다른 learning rate)
        self.optimizer = optimizer([
            {'params': self.model.encoder.parameters()},
            {'params': self.model.decoder.parameters(), 'lr': decoder_lr}
        ], lr=encoder_lr, weight_decay=weight_decay)

        # Scheduler 설정 (옵션)
        self.lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        self.ails = ails
        self.log = self.init_log()
        self.logging_step = 0
        self.start_epoch = start_epoch if start_epoch else 0

        os.makedirs("../data/weight", exist_ok=True)
        os.makedirs("../data/result_log", exist_ok=True)

    def init_log(self):
        # 학습 로그 초기화
        if self.one_channel:
            return {
                "command": "python main.py --train train --task damage --label all",
                "start_at_kst": 1,
                "end_at_kst": 1,
                "train_log": []
            }
        else:
            # COCO categories 정보 포함
            categories = {0: {'id': 0, 'name': 'Background'}}
            categories.update(self.train_dataset.coco.cats)
            return {
                "command": "python main.py --train train --task part --cls 16",
                "start_at_kst": 1,
                "end_at_kst": 1,
                "train_log": [],
                "category": categories
            }

    def get_dataloader(self):
        # DataLoader 생성
        def collate_fn(batch):
            return tuple(zip(*batch))

        train_loader = DataLoader(self.train_dataset, shuffle=True, num_workers=4,
                                  collate_fn=collate_fn, batch_size=self.batch_size)
        val_loader = DataLoader(self.val_dataset, shuffle=False, num_workers=4,
                                collate_fn=collate_fn, batch_size=self.batch_size)
        return train_loader, val_loader

    def train(self):
        print('--- start-training ---')
        self.log['start_at_kst'] = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S %Z%z')

        train_loader, val_loader = self.get_dataloader()
        self.model.to(self.device)

        best_mIoU = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print(f"[Epoch {epoch + 1}/{self.start_epoch + self.epochs}]")
            self.model.train()
            train_losses = []

            for step, (images, masks, _) in enumerate(train_loader):
                # 텐서 변환 및 GPU 이동
                images = torch.as_tensor(images, dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(masks, dtype=torch.long, device=self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    print(f"Step {step} - Loss: {loss.item():.4f}")
                train_losses.append(loss.item())

            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 로그에 현재 epoch 결과 추가
            self.log['train_log'].append({
                "epoch": epoch + 1,
                "train_loss": train_losses,
                "eval": {
                    "img": [],
                    "summary": {
                        "mIoU": 0.0,
                        "average Loss": 0.0,
                        "background IoU": 0.0,
                        "target IoU": 0.0,
                        "end_at_kst": 0
                    }
                }
            })

            # 검증 수행
            avrg_loss, mIoU, cls_IoU = self.validation(epoch, val_loader)
            self.logging_step += 1

            # 최고 성능 모델 저장
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                file_suffix = (
                    f"start:{self.log['start_at_kst']}_{epoch + 1}_epoch_IoU_{float(cls_IoU[1] * 100):.1f}"
                    if self.one_channel else
                    f"start:{self.log['start_at_kst']}_{epoch + 1}_epoch_IoU_{float(mIoU * 100):.1f}"
                )
                model_path = (
                    f"../data/weight/Unet_{self.ails}_label{self.label}_{file_suffix}"
                    if self.one_channel else
                    f"../data/weight/Unet_{self.ails}_{file_suffix}"
                )
                log_path = (
                    f"../data/result_log/[{self.ails}_label{self.label}]train_log.json"
                    if self.one_channel else
                    f"../data/result_log/[{self.ails}]train_log.json"
                )
                self.save_model(model_path)
                with open(log_path, "w") as f:
                    json.dump(self.log, f)

    def save_model(self, file_name):
        # 모델 가중치 저장
        file_name += '.pt'
        torch.save(self.model.state_dict(), file_name)
        print(f"[save_model] MODEL SAVED to {file_name}")

    def validation(self, epoch_idx, val_loader):
        n_class = self.n_class
        self.model.eval()
        total_loss = 0.0
        cnt = 0
        hist = np.zeros((n_class, n_class))

        with torch.no_grad():
            for step, (images, masks, img_ids) in enumerate(val_loader):
                images = torch.as_tensor(images, dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(masks, dtype=torch.long, device=self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                cnt += 1

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gt = masks.cpu().numpy()

                # 이미지별 IoU 계산
                for i, img_id in enumerate(img_ids):
                    h = add_hist(np.zeros((n_class, n_class)), gt[i], preds[i], n_class=n_class)
                    _, _, mIoU_sample, _, cls_IoU_sample = label_accuracy_score(h)
                    self.log["train_log"][self.logging_step]['eval']['img'].append(
                        {"img_id": img_id, "IoU": list(cls_IoU_sample)}
                    )

                # 전체 confusion matrix 누적
                hist = add_hist(hist, gt, preds, n_class=n_class)

            acc, acc_cls, mIoU, fwavacc, cls_IoU = label_accuracy_score(hist)
            avrg_loss = total_loss / cnt

            end_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S %Z%z')
            self.log["end_at_kst"] = end_time
            self.log["train_log"][self.logging_step]['eval']['summary'] = {
                "mIoU": float(mIoU),
                "average Loss": float(avrg_loss),
                "background IoU": float(cls_IoU[0]),
                "target IoU": float(cls_IoU[1]) if self.one_channel else list(cls_IoU[1:]),
                "end_at_kst": end_time
            }

            print(
                f"[Validation] epoch {epoch_idx + 1} | Avg Loss: {avrg_loss:.4f}, "
                f"mIoU: {mIoU:.4f}, BG IoU: {cls_IoU[0]:.4f}, "
                f"{'Target' if self.one_channel else 'Others'}: "
                f"{cls_IoU[1:] if not self.one_channel else cls_IoU[1]:.4f}"
            )
            return avrg_loss, mIoU, cls_IoU
