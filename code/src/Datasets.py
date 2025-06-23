import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A


class Datasets(Dataset):
    def __init__(self,
                 data_dir,
                 mode,
                 size,
                 label=None,
                 one_channel=False,
                 img_base_path=None,
                 transform=None):

        super().__init__()
        self.coco = COCO(data_dir)
        self.mode = mode
        self.label = label
        self.one_channel = one_channel
        self.img_base_path = img_base_path
        self.transform = transform
        self.size = size

        if self.img_base_path is None:
            raise ValueError("img_base_path를 꼭 전달해주세요!")

        if mode in ("train", "test"):
            self.img_ids = self.coco.getImgIds()
        else:
            self.img_ids = np.random.choice(self.coco.getImgIds(), 300, replace=False)

        self.resize = A.Compose([A.Resize(width=self.size, height=self.size)]) if self.size else None

    def __getitem__(self, index):
        image_id = int(self.img_ids[index])
        image_info = self.coco.loadImgs(image_id)[0]

        img_path = os.path.join(self.img_base_path, image_info["file_name"])
        image = cv2.imread(img_path)

        if image is None:
            print(f"[경고] 이미지 없음: {image_info['file_name']} → 건너뜀")
            return self.__getitem__((index + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros((image_info["height"], image_info["width"]))

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            if self.one_channel:
                for ann in anns:
                    if ann['category_id'] == self.label:
                        mask = np.maximum(self.coco.annToMask(ann), mask)
                mask = mask.astype(np.float32)
            else:
                for ann in anns:
                    pixel_val = ann['category_id'] + 1
                    mask = np.maximum(self.coco.annToMask(ann) * pixel_val, mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        elif self.resize:
            transformed = self.resize(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        image = image / 255.0
        image = image.transpose(2, 0, 1)

        return image, mask, image_info['file_name']

    def __len__(self):
        return len(self.img_ids)
