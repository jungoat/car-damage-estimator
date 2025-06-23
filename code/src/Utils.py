import os
import json
import glob
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


class COCOFormatter:
    def __init__(self, task, csv_path, img_dir, ann_dir, output_dir, labeling_scheme, file_column):
        self.task = task  # 'damage' or 'part'
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.output_dir = output_dir
        self.labeling_scheme = labeling_scheme
        self.file_column = file_column

    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def save_json(self, obj, path):
        with open(path, "w") as f:
            json.dump(obj, f)

    def process_subset(self, subset, alias):
        df = pd.read_csv(self.csv_path)
        if self.task == "damage":
            df = df[df.total_anns > 0]
        subset_list = df[df.dataset == subset][self.file_column].tolist()

        images = [os.path.join(self.img_dir, f.replace(".json", ".jpg")) for f in subset_list]
        annotations = [os.path.join(self.ann_dir, f.replace(".jpg", ".json")) for f in subset_list]

        if not annotations:
            print(f"[경고] '{self.ann_dir}' 폴더에 .json이 없습니다.")
            return

        base_json = self.load_json(annotations[0])
        base_json['images'] = []
        base_json['annotations'] = []

        base_json['categories'] = [
            {"id": i, "name": name} for i, name in enumerate(self.labeling_scheme)
        ]
        if self.task == "part":
            base_json['categories'].append({"id": len(self.labeling_scheme), "name": "etc"})

        img_id, ann_id = 0, 0

        for ann_file in annotations:
            if not os.path.exists(ann_file):
                print(f"[경고] '{ann_file}' 없음")
                continue

            ann_data = self.load_json(ann_file)
            if not ann_data.get("annotations"):
                continue

            img_id += 1
            ann_data['images']['id'] = img_id
            base_json['images'].append(ann_data['images'])

            for a in ann_data['annotations']:
                label = a.get(self.task, '')
                if label:
                    ann_id += 1
                    a['id'] = ann_id
                    a['image_id'] = img_id
                    a['category_id'] = self.labeling_scheme.index(label) if label in self.labeling_scheme else len(self.labeling_scheme)
                    a['segmentation'] = [a['segmentation']]
                    base_json['annotations'].append(a)

        print(f"{alias}: total images = {len(base_json['images'])}, total annotations = {len(base_json['annotations'])}")
        self.save_json(base_json, os.path.join(self.output_dir, f"{alias}.json"))


# ============================
# 평가용 함수 / focal loss
# ============================
def label_accuracy_score(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = 0.0 if np.isnan(iu).all() else np.nanmean(iu)
    cls_iu = np.where(np.isnan(iu), -1, iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, cls_iu

def _fast_hist(true, pred, n_class):
    mask = (true >= 0) & (true < n_class)
    return np.bincount(n_class * true[mask].astype(int) + pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)

def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        targets = targets.long()
        at = self.alpha.gather(0, targets.view(-1))
        pt = torch.exp(-ce_loss)
        return (at * (1 - pt) ** self.gamma * ce_loss).mean()


# ============================
# CLI 실행
# ============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--make_cocoformat', action='store_true')
    parser.add_argument('--task', choices=['damage', 'part'])
    args = parser.parse_args()

    if not args.make_cocoformat:
        exit()

    if args.task == 'damage':
        csv_path = "/home/lhh5785/.../damage_labeling.csv"
        img_dir = "/home/lhh5785/.../images/damage"
        ann_dir = "/home/lhh5785/.../labels/damage_labeled/damage"
        output_dir = "/home/lhh5785/.../labels/damage_labeled"
        labeling_scheme = ["Scratched", "Separated", "Crushed", "Breakage"]
        file_column = "index"
    else:
        csv_path = "/home/lhh5785/.../part_labeling.csv"
        img_dir = "/home/lhh5785/.../images/damage_part"
        ann_dir = "/home/lhh5785/.../labels/part_labeled/damage_part"
        output_dir = "/home/lhh5785/.../labels/part_labeled"
        labeling_scheme = ["Front bumper", "Rear bumper", "Front fender(R)", "Front fender(L)", "Rear fender(R)", "Trunk lid", "Bonnet", "Rear fender(L)", "Rear door(R)", "Head lights(R)", "Head lights(L)", "Front Wheel(R)", "Front door(R)", "Side mirror(R)"]
        file_column = "img_id"

    for subset in ["train", "val", "test"]:
        alias = f"{args.task}_{subset}"
        formatter = COCOFormatter(
            task=args.task,
            csv_path=csv_path,
            img_dir=img_dir,
            ann_dir=ann_dir,
            output_dir=output_dir,
            labeling_scheme=labeling_scheme,
            file_column=file_column
        )
        formatter.process_subset(subset, alias)

    print(f"DONE {args.task}!")
