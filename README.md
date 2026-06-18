# CNN-Based-Deepfake-Detection-Model

본 프로젝트는 CNN 기반 이미지 분류 모델을 활용해 Real 얼굴 이미지와 AI 기반 얼굴 조작 이미지(Fake)를 구분하는 딥페이크 탐지 모델이다.

최종 모델은 **EfficientNet-B4**를 backbone으로 사용한다.
원본 RGB 이미지와 Artifact Map 이미지를 각각 학습한 뒤, 두 모델의 Fake probability를 평균하는 **RGB + Artifact Map Ensemble** 방식을 적용한다.

---

## 1. 프로젝트 개요

딥페이크 기술의 품질이 높아지면서 가짜 뉴스, 얼굴 도용, 명예훼손, 사생활 침해와 같은 문제가 증가하고 있다.
특히 최근 생성·편집된 얼굴 이미지는 사람 눈으로 Real/Fake를 구분하기 어려운 경우가 많아 이미지 기반 자동 탐지 모델의 필요성이 커진다.

본 프로젝트의 목표는 다음과 같다.

* Real 얼굴 이미지와 AI 기반 얼굴 조작 이미지(Fake)를 구분한다.
* RGB 이미지와 Artifact Map을 함께 활용해 탐지 성능을 개선한다.
* CNN 기반 모델과 Transformer 기반 모델을 비교하여 최종 모델을 선정한다.

---

## 2. 데이터셋 구성

본 프로젝트에서는 Real 이미지 500장, Fake 이미지 500장으로 총 1000장의 이미지 데이터셋을 구성한다.

| Class | Count |
| ----- | ----: |
| Real  |   500 |
| Fake  |   500 |
| Total |  1000 |

Train과 Validation은 8:2 비율로 분할한다.

| Split      | Count |
| ---------- | ----: |
| Train      |   800 |
| Validation |   200 |

데이터셋 구조는 다음과 같다.

```text
data/
├── train/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```

이미지 데이터셋은 용량 및 저작권 문제로 GitHub에는 포함하지 않는다.
실행 전 위 폴더 구조에 맞게 데이터를 배치한다.

---

## 3. 제안 방법

### 3.1 RGB Image 기반 탐지

RGB 모델은 원본 얼굴 이미지를 입력으로 사용한다.

```text
Original RGB Image → EfficientNet-B4 → Real / Fake
```

RGB 이미지는 얼굴의 전체적인 구조, 색상, 조명 정보를 포함하고 있어 기본적인 Real/Fake 분류에 사용한다.

---

### 3.2 Artifact Map 기반 탐지

Artifact Map은 원본 이미지와 Gaussian Blur 이미지의 차이를 계산하여 생성한다.

```text
Artifact Map = | Original Image - Gaussian Blurred Image |
```

Gaussian Blur를 적용하면 얼굴의 전체 구조, 색상, 조명과 같은 저주파 정보는 남고, 피부 질감, 경계선, 노이즈와 같은 고주파 정보는 약해진다.

따라서 원본 이미지에서 Gaussian Blur 이미지를 빼면 고주파 성분이 강조된 Artifact Map을 얻을 수 있다.

```text
Original Image → Gaussian Blur → Difference Image → Artifact Map
```

Artifact Map은 사람이 직접 Fake 여부를 판별하기 위한 이미지가 아니라, CNN 모델이 미세한 질감, 경계선, 노이즈 패턴을 보조적으로 학습하도록 만든 입력으로 사용한다.

---

### 3.3 RGB + Artifact Map Ensemble

RGB 모델과 Artifact Map 모델은 각각 따로 학습한다.

이후 두 모델의 Fake probability를 평균하여 최종 Fake probability를 계산한다.

```text
Final Fake Probability = (RGB Fake Probability + Artifact Fake Probability) / 2
```

최종 판정 기준은 threshold 0.5로 설정한다.

```text
Fake Probability >= 0.5 → FAKE
Fake Probability < 0.5  → REAL
```

앙상블을 사용한 이유는 RGB 이미지와 Artifact Map이 서로 다른 정보를 담고 있기 때문이다.

* RGB Model: 얼굴의 전체 구조, 색상, 조명 정보를 학습한다.
* Artifact Model: 피부 질감, 얼굴 경계선, 노이즈와 같은 고주파 성분을 학습한다.

따라서 두 모델의 결과를 결합하면 단일 입력 모델보다 더 안정적인 판단을 가능하게 한다.

<p align="center">
<img width="500" height="400" alt="스크린샷_18-6-2026_162831_" src="https://github.com/user-attachments/assets/75ae25f6-022a-4ebf-9499-0d88cf38094c" />
</p>

---

## 4. 모델 선정 과정

본 프로젝트에서는 최종 backbone을 선정하기 위해 먼저 EfficientNet 계열 모델을 비교한다.
EfficientNet은 B0부터 B4까지 모델 크기와 입력 해상도가 단계적으로 확장되는 구조이기 때문에, 딥페이크 탐지에 적합한 모델 규모를 확인하기 위한 후보 모델로 사용한다.

### 4.1 EfficientNet 계열 1차 실험

먼저 EfficientNet-B0부터 EfficientNet-B4까지 동일한 입력 해상도인 224×224 조건에서 성능을 비교한다.
이 실험은 모델 크기가 커질수록 성능이 항상 좋아지는지 확인하기 위한 1차 비교 실험이다.

| Model           | Input Size | Purpose                            |
| --------------- | ---------: | ---------------------------------- |
| EfficientNet-B0 |    224×224 | 가장 작은 EfficientNet 모델 기준 성능을 확인한다. |
| EfficientNet-B1 |    224×224 | B0보다 확장된 모델 성능을 확인한다.              |
| EfficientNet-B2 |    224×224 | 중간 규모 모델 성능을 확인한다.                 |
| EfficientNet-B3 |    224×224 | 더 큰 EfficientNet 계열 성능을 확인한다.      |
| EfficientNet-B4 |    224×224 | 최종 후보 모델의 동일 해상도 조건 성능을 확인한다.      |

실험 결과, EfficientNet 계열에서 모델 크기가 커진다고 해서 224×224 입력 조건에서 성능이 항상 향상되지는 않음을 확인한다.
이를 통해 단순히 모델 크기만 비교하는 것이 아니라, 각 모델에 적합한 입력 해상도도 함께 고려해야 한다고 판단한다.

| Model           |    Size | Accuracy |
| --------------- | ------: | -------: |
| EfficientNet-B0 | 224×224 |   76.55% |
| EfficientNet-B1 | 224×224 |   75.68% |
| EfficientNet-B2 | 224×224 |   73.80% |
| EfficientNet-B3 | 224×224 |   71.24% |
| EfficientNet-B4 | 224×224 |   67.70% |
| EfficientNet-B5 | 224×224 |   57.70% |


<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/06124e41-dcd7-4264-a7f8-ae63f042e151" />


---

### 4.2 권장 입력 해상도 적용 실험

EfficientNet은 모델별 권장 입력 해상도가 존재한다.
특히 EfficientNet-B4는 380×380 입력 해상도를 권장하기 때문에, B4 모델에 380×380 해상도를 적용하여 추가 실험을 진행한다.

| Model           |    Size | Accuracy |
| --------------- | ------: | -------: |
| EfficientNet-B4 | 224×224 |   67.70% |
| EfficientNet-B4 | 380×380 |   85.90% |
| EfficientNet-B5 | 224×224 |   57.70% |
| EfficientNet-B5 | 456×456 |   71.40% |

권장 해상도 적용 후 EfficientNet-B4의 성능이 향상되는 것을 확인한다.
따라서 본 프로젝트에서는 얼굴의 미세한 질감, 경계선, 노이즈와 같은 정보를 더 잘 반영할 수 있는 EfficientNet-B4를 최종 backbone으로 선정한다.

---

### 4.3 최종 비교 모델 선정

EfficientNet-B4를 최종 후보 모델로 선정한 뒤, 다른 구조의 모델들과 성능을 비교하기 위해 총 4개의 모델을 실험 대상으로 선정한다.

| Model             | 선정 이유                                        |
| ----------------- | -------------------------------------------- |
| EfficientNet-B4   | 권장 해상도 적용 후 성능이 향상되어 최종 backbone 후보로 선정한다.   |
| MobileNetV3-Large | 경량 CNN 모델로, 실제 서비스 적용 가능성 비교를 위해 선정한다.       |
| ResNet-50         | 이미지 분류에서 널리 사용되는 전통적인 CNN baseline으로 선정한다.   |
| ViT-B/16          | CNN과 다른 Transformer 기반 이미지 모델과의 비교를 위해 선정한다. |

최종적으로 EfficientNet-B4는 RGB 입력과 Artifact Map 입력 모두에서 안정적인 성능을 보였고, RGB + Artifact Map 앙상블에서도 가장 높은 정확도를 기록한다.
따라서 본 프로젝트의 최종 모델은 **EfficientNet-B4 기반 RGB + Artifact Map Ensemble 구조**로 선정한다.

MobileNetV3-Large는 경량 모델로서 높은 성능을 보였으나, 본 프로젝트에서는 속도보다 최종 정확도와 입력별 안정성을 더 중요하게 판단한다.
ResNet-50과 ViT-B/16은 이번 데이터셋 및 학습 조건에서는 성능이 낮게 나타난다. 특히 ViT-B/16은 모든 이미지를 Fake로 예측하는 클래스 쏠림 현상이 발생하여 추가 검증이 필요한 결과로 해석한다.

---

## 5. 학습 설정

| Parameter            | Value               |
| -------------------- | ------------------- |
| Backbone             | EfficientNet-B4     |
| Input Size           | 380 × 380           |
| Batch Size           | 8                   |
| Epochs               | 10                  |
| Early Stopping       | patience 4          |
| Checkpoint Criterion | Min Validation Loss |

Min Validation Loss는 validation loss가 가장 낮게 나온 시점의 모델을 checkpoint로 저장한다는 의미이다.
즉, 마지막 epoch의 모델을 사용하는 것이 아니라 검증 데이터에서 오차가 가장 작았던 모델을 최종 평가에 사용한다.

---

## 6. 실행 방법

### 6.1 RGB 모델 학습

본 프로젝트는 하나의 `config.yaml` 파일에서 `input_mode` 값을 변경하여 RGB 모델과 Artifact Map 모델을 각각 학습한다.

RGB 모델을 학습할 때는 `config.yaml`을 다음과 같이 설정한다.

```yaml
input_mode: "rgb"
checkpoint_filename: "rgb_model"
```

이후 학습을 실행한다.

```bash
python main_trainer.py
```

학습 완료 후 다음 checkpoint를 저장한다.

```text
models/rgb_model.ckpt
```

---

### 6.2 Artifact Map 모델 학습

Artifact Map 모델을 학습할 때는 `config.yaml`을 다음과 같이 설정한다.

```yaml
input_mode: "artifact"
checkpoint_filename: "artifact_map_model"
```

이후 학습을 실행한다.

```bash
python main_trainer.py
```

학습 완료 후 다음 checkpoint를 저장한다.

```text
models/artifact_map_model.ckpt
```

주의: `classify.py`의 기본 설정은 `models/rgb_model.ckpt`와 `models/artifact_map_model.ckpt`를 불러오도록 되어 있다.
따라서 checkpoint 이름을 위와 같이 맞추는 것을 권장한다.

---

### 6.3 단일 이미지 예측

`classify.py`는 학습된 RGB 모델과 Artifact Map 모델을 불러와 단일 이미지에 대한 Real/Fake 예측을 수행한다.

기본적으로 다음 두 checkpoint 파일을 사용한다.

```text
models/rgb_model.ckpt
models/artifact_map_model.ckpt
```

따라서 두 checkpoint 파일이 위 경로에 존재한다면, 예측할 이미지 경로만 입력하여 실행한다.

```bash
python classify.py path/to/image.jpg
```

예를 들어 `test.jpg` 이미지를 예측하려면 다음과 같이 실행한다.

```bash
python classify.py test.jpg
```

`classify.py`는 RGB 모델의 Fake probability와 Artifact Map 모델의 Fake probability를 각각 계산한 뒤, 두 값을 평균하여 최종 Fake probability를 구한다.

```text
Final Fake Probability = (RGB Fake Probability + Artifact Fake Probability) / 2
```

출력 예시는 다음과 같다.

```text
Prediction: REAL
[Ensemble Result]
Real: 0.870 | Fake: 0.130
```

Fake probability가 0.5 이상이면 FAKE, 0.5 미만이면 REAL로 분류한다.

만약 checkpoint 파일명이 다르거나 다른 폴더에 저장되어 있다면, 아래와 같이 직접 checkpoint 경로를 지정할 수 있다.

```bash
python classify.py path/to/image.jpg --rgb_model models/rgb_model.ckpt --artifact_model models/artifact_map_model.ckpt
```



<img width="200" height="60" alt="image" src="https://github.com/user-attachments/assets/b32d7131-563e-4937-b243-d723ca4d25a1" />

---

## 7. 실험 결과

### 7.1 RGB / Artifact 단일 입력 성능 비교

RGB 입력과 Artifact Map 입력을 각각 따로 사용하여 모델별 성능을 비교한다.


<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/af6f61d2-d1a3-4985-b23b-0c15b7a85809" />


| Model             | RGB Accuracy | Artifact Accuracy |
| ----------------- | -----------: | ----------------: |
| EfficientNet-B4   |       90.00% |            91.50% |
| MobileNetV3-Large |       91.50% |            87.00% |
| ResNet-50         |       52.00% |            54.00% |
| ViT-B/16          |       50.00% |            50.00% |

결과적으로 RGB 입력에서는 MobileNetV3와 EfficientNet-B4가 높은 성능을 보였고, Artifact Map 입력에서는 EfficientNet-B4가 가장 안정적인 성능을 보인다.

---

### 7.2 RGB + Artifact Map Ensemble 성능 비교

RGB 모델과 Artifact Map 모델의 Fake probability를 평균하여 최종 예측을 수행한다.

| Model             | Ensemble Accuracy |
| ----------------- | ----------------: |
| EfficientNet-B4   |            93.50% |
| MobileNetV3-Large |            92.50% |
| ResNet-50         |            55.50% |
| ViT-B/16          |            50.00% |


<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/7f90c57c-a8bd-4592-826c-7bf28cac4b65" />


EfficientNet-B4는 최종 앙상블 정확도 93.50%로 가장 높은 성능을 보인다.
따라서 최종 모델은 EfficientNet-B4 기반 RGB + Artifact Map Ensemble 구조로 선정한다.

---

### 7.3 Threshold 및 Confusion Matrix 분석

최종 EfficientNet-B4 Ensemble 모델은 RGB 모델과 Artifact Map 모델의 Fake probability를 평균하여 최종 Fake probability를 계산한다.
이후 threshold 값을 기준으로 Real/Fake를 판정한다.

```text
Final Fake Probability = (RGB Fake Probability + Artifact Fake Probability) / 2
```

```text
Fake Probability >= Threshold → FAKE
Fake Probability < Threshold  → REAL
```

본 프로젝트에서는 threshold를 0.5로 설정한다.
threshold 0.5는 이진 분류에서 가장 기본적인 판정 기준이며, Fake probability가 50% 이상이면 Fake로 판단한다는 의미이다.

다만 단순히 기본값이라서 0.5를 사용한 것이 아니라, threshold 값을 변화시키며 Accuracy, Precision, Recall, F1-score, FP, FN 변화를 비교한다.

| Threshold | Accuracy | Precision | Recall | F1-score | FP | FN |
| --------: | -------: | --------: | -----: | -------: | -: | -: |
|      0.50 |   93.50% |     0.958 |  0.910 |    0.933 |  4 |  9 |
|      0.55 |   93.00% |     0.978 |  0.880 |    0.926 |  2 | 12 |
|      0.60 |   93.50% |     0.989 |  0.880 |    0.931 |  1 | 12 |
|      0.65 |   91.00% |     0.988 |  0.830 |    0.902 |  1 | 17 |
|      0.70 |   91.00% |     1.000 |  0.820 |    0.901 |  0 | 18 |

threshold를 높이면 Fake로 판단하는 기준이 엄격해지기 때문에 False Positive는 줄어드는 경향을 보인다.
예를 들어 threshold 0.70에서는 Precision이 1.000으로 가장 높고 FP가 0개였지만, Fake 이미지를 Real로 잘못 판단한 FN이 18개까지 증가한다. 즉, Fake를 놓치는 경우가 많아진다.

반면 threshold 0.50에서는 Accuracy가 93.50%로 가장 높았고, F1-score도 0.933으로 가장 높게 나타난다.
또한 Recall이 0.910으로 다른 threshold보다 높아 Fake 이미지를 놓치는 비율이 상대적으로 낮다.

따라서 본 프로젝트에서는 Precision과 Recall의 균형을 가장 잘 보여주는 F1-score를 기준으로 threshold 0.5를 최종 판정 기준으로 선택한다.

<img width="600" height="350" alt="Threshold Analysis" src="https://github.com/user-attachments/assets/86c8a083-03f6-4b1a-8adb-342f3fc3c174" />

최종 EfficientNet-B4 Ensemble 모델의 threshold 0.5 기준 성능은 다음과 같다.

| Metric       |   Result |
| ------------ | -------: |
| Accuracy     |   93.50% |
| Precision    |    0.958 |
| Recall       |    0.910 |
| F1-score     |    0.933 |
| Real Correct | 96 / 100 |
| Fake Correct | 91 / 100 |

Confusion Matrix는 다음과 같다.

|             | Pred Real | Pred Fake |
| ----------- | --------: | --------: |
| Actual Real |        96 |         4 |
| Actual Fake |         9 |        91 |

Confusion Matrix 기준으로 Real 이미지 100장 중 96장을 Real로 올바르게 분류하고, Fake 이미지 100장 중 91장을 Fake로 올바르게 분류한다.
즉, 전체 200장의 validation 이미지 중 187장을 올바르게 분류하여 최종 Accuracy는 93.50%로 나타난다.

<img width="600" height="350" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/570ee874-4bba-48a8-956d-f6a9b06a6434" />

---

## 8. 결과 해석

EfficientNet-B4는 RGB 입력과 Artifact Map 입력 모두에서 안정적인 성능을 보였고, 최종 앙상블에서도 가장 높은 정확도를 기록한다.

MobileNetV3-Large는 경량 모델임에도 높은 성능을 보였으나, Artifact Map 입력에서 EfficientNet-B4보다 낮은 성능을 보인다.
따라서 본 프로젝트에서는 속도보다 탐지 성능과 입력별 안정성을 우선하여 EfficientNet-B4를 최종 모델로 선정한다.

ResNet-50은 전통적인 CNN baseline으로 사용했지만, 이번 데이터셋 조건에서는 성능이 낮게 나타난다.

ViT-B/16은 모든 이미지를 Fake로 예측하는 클래스 쏠림 현상이 발생한다.
이는 데이터셋 규모, 입력 해상도, fine-tuning 설정 차이의 영향을 받은 것으로 해석했으며, 추가 검증이 필요한 결과로 판단한다.

---

## 9. 한계점

### 9.1 데이터셋 유형

본 프로젝트의 Fake 데이터는 AI 기반 표정 조작 및 얼굴 편집 이미지 중심으로 구성된다.
따라서 모든 유형의 딥페이크 일반 탐지보다는 face manipulation 이미지 탐지로 해석할 필요가 있다.

### 9.2 Artifact Map의 한계

Artifact Map은 고주파 성분을 강조하지만, 조명, 화장, 피부 질감, 압축 노이즈도 함께 반영될 수 있다.

### 9.3 데이터 편향 가능성

로고, 워터마크, 배경 차이가 특정 클래스에만 존재할 경우 모델이 얼굴 조작 특징이 아닌 외부 요소에 영향을 받을 수 있다.

---

## 10. 향후 과제

### 10.1 데이터 다양화

Face Swap, 표정 조작, GAN 생성 등 다양한 얼굴 조작 유형을 추가하여 모델의 일반화 성능을 향상시킨다.

### 10.2 전처리 고도화

얼굴 crop 및 alignment를 적용하여 얼굴 영역 중심의 탐지 성능을 개선한다.

### 10.3 분석 및 서비스 확장

주파수 영역 분석, 영상 기반 temporal artifact 분석, 웹 서비스 연동 등으로 확장한다.
