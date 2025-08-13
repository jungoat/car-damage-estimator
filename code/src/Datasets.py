import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A


class Datasets(Dataset):
    def __init__(self, data_dir, mode, size, label=None, one_channel=False,
                 img_base_path=None, transform=None):
        super().__init__()
        self.coco = COCO(data_dir)  # COCO format annotation 불러오기
        self.mode = mode            # 'train', 'val', 'test'
        self.size = size
        self.label = label
        self.one_channel = one_channel  # True면 단일 클래스 segmentation
        self.img_base_path = img_base_path
        self.transform = transform

        if self.img_base_path is None:
            raise ValueError("img_base_path 경로가 지정되지 않음")

        # 사용할 이미지 ID 목록 결정
        if mode in ("train", "test"):
            self.img_ids = self.coco.getImgIds()
        else:
            # val/test 샘플 수 제한 (여기서는 300개 랜덤 추출)
            self.img_ids = np.random.choice(self.coco.getImgIds(), 300, replace=False)

        # 리사이즈 transform 설정
        self.resize = A.Compose([A.Resize(width=self.size, height=self.size)]) if self.size else None

    def __getitem__(self, index):
        # 이미지 정보 로드
        image_id = int(self.img_ids[index])
        image_info = self.coco.loadImgs(image_id)[0]

        # 이미지 경로 및 읽기
        img_path = os.path.join(self.img_base_path, image_info["file_name"])
        image = cv2.imread(img_path)

        # 이미지 없으면 다음 인덱스 로드
        if image is None:
            print(f" 이미지 없음: {image_info['file_name']} -> pass")
            return self.__getitem__((index + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 초기 mask (전부 0)
        mask = np.zeros((image_info["height"], image_info["width"]))

        # 학습/검증 시에만 mask 생성
        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            if self.one_channel:
                # 단일 클래스 mask
                for ann in anns:
                    if ann['category_id'] == self.label:
                        mask = np.maximum(self.coco.annToMask(ann), mask)
                mask = mask.astype(np.float32)
            else:
                # 다중 클래스 mask (배경=0, 클래스는 category_id+1)
                for ann in anns:
                    pixel_val = ann['category_id'] + 1
                    mask = np.maximum(self.coco.annToMask(ann) * pixel_val, mask)

        # transform 적용 (있으면 우선 적용)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        elif self.resize:
            transformed = self.resize(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # [0,1] 범위로 정규화 후 (C,H,W) 형태로 변환
        image = image / 255.0
        image = image.transpose(2, 0, 1)

        return image, mask, image_info['file_name']

    def __len__(self):
        return len(self.img_ids)
