# 객체 분할 기반 차량 손상 진단 시스템 개발  

본 프로젝트는 차량 외관 손상 이미지를 입력받아 손상 부위를 분할(Segmentation)하고, 해당 부위에 대한 수리비를 예측하는 시스템입니다.  
AI-Hub 차량 손상 이미지와 실제 부품 수리비 데이터를 기반으로 모델을 학습하여 실시간 견적 추론을 가능하게 합니다.

🏆 2025 공학대학 캡스톤디지인 금상 수상

---

##  프로젝트 포스터

![Project Poster](Capstone_Poster.jpg)

---

##  Directory Structure

```
cap_code/
├── src/             # Model definitions and training/evaluation logic
│   ├── Models/
│   ├── Train.py
│   ├── Evaluation.py
│   └── ...
├── data/            # Images, labels, pretrained weights, etc.
│   ├── Dataset/
│   ├── datainfo/
│   └── weight/
├── test_infer/      # Script for single image inference
├── result_img/      # Directory to save inference results
├── poster.jpg       # Project poster image
└── README.md        # Project description
```

---

##  How to Run

### 1. Environment Setup

```bash
conda create -n damage python=3.9
conda activate damage
pip install -r requirements.txt
```

### 2. Training

```bash
python main.py --task [TASK_NAME] --label [CLASS_LABEL] --dataset train
```

예시:

```bash
python main.py --task seg --label C_ER_S --dataset train
```

### 3. Inference

```bash
python test_infer.py
```

---

##  Model

- Backbone: DeepLabV3+ (ResNet-50 기반)
- Loss: Focal Loss + Lovasz Softmax
- Label Format: COCO-style JSON instance segmentation
- Task: Damage segmentation & repair cost prediction
