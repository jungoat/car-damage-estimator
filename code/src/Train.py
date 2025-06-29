import os
import json
import datetime
from pytz import timezone

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

        self.model = model.model
        self.n_class = n_class
        self.criterion = criterion
        self.device = device

        self.epochs = epochs
        self.batch_size = batch_size
        self.label = label
        self.one_channel = (label is not None)

        self.train_dataset = Datasets(train_dir, 'train', size, label, self.one_channel, train_img_base_path, transform)
        self.val_dataset = Datasets(val_dir, 'val', size, label, self.one_channel, val_img_base_path)

        self.optimizer = optimizer([
            {'params': self.model.encoder.parameters()},
            {'params': self.model.decoder.parameters(), 'lr': decoder_lr}
        ], lr=encoder_lr, weight_decay=weight_decay)

        self.lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        self.ails = ails
        self.log = self.init_log()
        self.logging_step = 0
        self.start_epoch = start_epoch if start_epoch else 0

        os.makedirs("../data/weight", exist_ok=True)
        os.makedirs("../data/result_log", exist_ok=True)

    def init_log(self):
        if self.one_channel:
            return {
                "command": "python main.py --train train --task damage --label all",
                "start_at_kst": 1,
                "end_at_kst": 1,
                "train_log": []
            }
        else:
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
        def collate_fn(batch):
            return tuple(zip(*batch))

        train_loader = DataLoader(self.train_dataset, shuffle=True, num_workers=4,
                                  collate_fn=collate_fn, batch_size=self.batch_size)
        val_loader = DataLoader(self.val_dataset, shuffle=False, num_workers=4,
                                collate_fn=collate_fn, batch_size=self.batch_size)
        return train_loader, val_loader

    def train(self):
        print(f'--- start-training ---')
        self.log['start_at_kst'] = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S %Z%z')

        train_loader, val_loader = self.get_dataloader()
        self.val_data_loader = val_loader
        self.model.to(self.device)

        best_mIoU = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            print(f"[Epoch {epoch + 1}/{self.start_epoch + self.epochs}]")
            self.model.train()
            train_losses = []

            for step, (images, masks, _) in enumerate(train_loader):
                images = torch.tensor(images).float().to(self.device)
                masks = torch.tensor(masks).long().to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    print(f"Step {step} - Loss: {loss.item()}")
                train_losses.append(loss.item())

            if self.lr_scheduler:
                self.lr_scheduler.step()

            base_form = {
                "epoch": epoch + 1,
                "train_loss": train_losses,
                "eval": {
                    "img": [],
                    "summary": {
                        "Imou": 0.0,
                        "Average Loss": 0.0,
                        "background IoU": 0.0,
                        "target IoU": 0.0,
                        "end_at_kst": 0
                    }
                }
            }
            self.log['train_log'].append(base_form)

            avrg_loss, mIoU, cls_IoU = self.validation(epoch, val_loader)
            self.logging_step += 1

            if mIoU > best_mIoU:
                best_mIoU = mIoU
                file_suffix = f"start:{self.log['start_at_kst']}_{epoch + 1}_epoch_IoU_{float(cls_IoU[1] * 100):.1f}" if self.one_channel else f"start:{self.log['start_at_kst']}_{epoch + 1}_epoch_IoU_{float(mIoU * 100):.1f}"
                model_path = f"../data/weight/Unet_{self.ails}_label{self.label}_{file_suffix}" if self.one_channel else f"../data/weight/Unet_{self.ails}_{file_suffix}"
                log_path = f"../data/result_log/[{self.ails}_label{self.label}]train_log.json" if self.one_channel else f"../data/result_log/[{self.ails}]train_log.json"
                self.save_model(model_path)
                with open(log_path, "w") as f:
                    json.dump(self.log, f)

    def save_model(self, file_name):
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
                images = torch.tensor(images).float().to(self.device)
                masks = torch.tensor(masks).long().to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                cnt += 1

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                gt = masks.cpu().numpy()

                for i, img_id in enumerate(img_ids):
                    h = np.zeros((n_class, n_class))
                    h = add_hist(h, gt[i], preds[i], n_class=n_class)
                    _, _, mIoU_sample, _, cls_IoU_sample = label_accuracy_score(h)
                    self.log["train_log"][self.logging_step]['eval']['img'].append({"img_id": img_id, "IoU": list(cls_IoU_sample)})

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

            print(f"[Validation] epoch {epoch_idx + 1} | Avg Loss: {avrg_loss:.4f}, mIoU: {mIoU:.4f}, BG IoU: {cls_IoU[0]:.4f}, {'Target' if self.one_channel else 'Others'}: {cls_IoU[1:] if not self.one_channel else cls_IoU[1]:.4f}")
            return avrg_loss, mIoU, cls_IoU
