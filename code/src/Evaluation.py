import os
import json
import datetime
from pytz import timezone
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from Datasets import Datasets
from Utils import label_accuracy_score, add_hist


class Evaluation:
    def __init__(self,
                 eval_dir,        # 평가 데이터 JSON 경로
                 size,            # 입력 이미지 크기
                 model,           # 모델 객체
                 weight_paths,    # 모델 가중치 경로 리스트 (multi-model 지원)
                 device,
                 batch_size,
                 ails,            # 로그 저장 경로
                 criterion,       # 손실 함수
                 img_base_path):  # 이미지 루트 경로

        self.eval_dir = eval_dir
        self.img_base_path = img_base_path
        self.size = size
        self.model = model
        self.weight_paths = weight_paths
        self.device = device
        self.batch_size = batch_size
        self.ails = ails
        self.criterion = criterion

        # 모델 여러 개인지 여부
        self.multi_model = len(weight_paths) > 1
        self.one_channel = self.multi_model
        self.n_class = 2 if self.multi_model else 16

        # 평가 로그 초기화
        self.log = {
            "command": f"python main.py --eval y --task {'damage' if self.multi_model else 'part'} --dataset val",
            "start_at_kst": 1,
            "end_at_kst": 1,
            "evaluation": []
        }

        self.logging_step = 0

    def get_dataloader(self, dataset):
        # DataLoader 생성 (collate_fn: batch 구성 방식)
        def collate_fn(batch):
            return tuple(zip(*batch))
        return DataLoader(dataset=dataset, shuffle=False, num_workers=4,
                          collate_fn=collate_fn, batch_size=self.batch_size)

    def evaluation(self):
        # 평가 시작 시간 기록
        self.log['start_at_kst'] = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S %Z%z')

        if self.multi_model:
            # damage 4종 모델 각각 평가
            labels_name = ["Scratched", "Separated", "Crushed", "Breakage"]
            for label_idx, weight_path in enumerate(self.weight_paths):
                print(f"[Evaluation] Damage Label = {labels_name[label_idx]}")
                self.log['evaluation'].append({"label": labels_name[label_idx], "eval": {"img": [], "summary": {}}})

                model_loaded = self.load_model(self.model, weight_path)
                dataset = Datasets(self.eval_dir, 'val', self.size, label_idx, True, self.img_base_path)
                self.log['category'] = dataset.coco.cats
                loader = self.get_dataloader(dataset)

                self.validation(model_loaded, label_idx, loader)

                with open(self.ails, "w") as f:
                    json.dump(self.log, f)

                torch.cuda.empty_cache()
                del loader, dataset, model_loaded

        else:
            # part 16클래스 모델 평가
            print("[Evaluation] Part (16-classes) Model")
            self.log['evaluation'].append({"eval": {"img": [], "summary": {}}})

            model_loaded = self.load_model(self.model, self.weight_paths[0])
            dataset = Datasets(self.eval_dir, 'val', self.size, None, False, self.img_base_path)
            self.log['category'] = dataset.coco.cats
            loader = self.get_dataloader(dataset)

            self.validation(model_loaded, 0, loader)

            with open(self.ails, "w") as f:
                json.dump(self.log, f)

        # 평가 종료 시간 기록
        self.log['end_at_kst'] = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S %Z%z')

    def load_model(self, model, weight_path):
        # 모델 로드 (모델 구조에 따라 접근 경로 다름)
        model = model.to(self.device)
        try:
            model.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            return model.model
        except:
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            return model

    def validation(self, model_loaded, label_idx, data_loader):
        model_loaded.eval()
        total_loss = 0.0
        cnt = 0
        hist = np.zeros((self.n_class, self.n_class))

        with torch.no_grad():
            for step, (images, masks, img_ids) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
                images = torch.as_tensor(images, dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(masks, dtype=torch.long, device=self.device)

                outputs = model_loaded(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                cnt += 1

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gt = masks.cpu().numpy()

                # 이미지별 IoU 기록
                for i, img_id in enumerate(img_ids):
                    h = add_hist(np.zeros((self.n_class, self.n_class)), gt[i], preds[i], self.n_class)
                    _, _, mIoU_sample, _, cls_IoU_sample = label_accuracy_score(h)
                    img_log = {"img_id": img_id, "IoU": list(cls_IoU_sample)}
                    if self.multi_model:
                        self.log['evaluation'][label_idx]['eval']['img'].append(img_log)
                    else:
                        self.log['evaluation'][0]['eval']['img'].append(img_log)

                # 전체 confusion matrix 누적
                hist = add_hist(hist, gt, preds, self.n_class)

        # 전체 통계 계산
        acc, acc_cls, mIoU, fwavacc, cls_IoU = label_accuracy_score(hist)
        avrg_loss = total_loss / cnt
        end_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S %Z%z')

        summary = {
            "mIoU": float(mIoU),
            "average Loss": float(avrg_loss),
            "background IoU": float(cls_IoU[0]),
            "target IoU": float(cls_IoU[1]) if self.one_channel else [float(x) for x in cls_IoU[1:]],
            "end_at_kst": end_time
        }

        if self.multi_model:
            self.log['evaluation'][label_idx]['eval']['summary'] = summary
        else:
            self.log['evaluation'][0]['eval']['summary'].update(summary)

        print(f"Val {'damage label='+str(label_idx) if self.multi_model else 'part (16-cls)'} "
              f"Loss={avrg_loss:.4f}, mIoU={mIoU:.4f}, BG={cls_IoU[0]:.4f}, "
              f"Target={cls_IoU[1] if self.one_channel else cls_IoU[1:]}")
        return avrg_loss, mIoU, cls_IoU
