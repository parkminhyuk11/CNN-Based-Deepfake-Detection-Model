# CNN-Based-Deepfake-Detection-Model



본 프로젝트는 CNN 기반 이미지 분류 모델을 활용해 Real 얼굴 이미지와 AI 기반 얼굴 조작 이미지(Fake)를 구분하는 딥페이크 탐지 모델.

최종 모델은 **EfficientNet-B4**를 backbone으로 사용.
원본 RGB 이미지와 Artifact Map 이미지를 각각 학습한 뒤, 두 모델의 Fake probability를 평균하는 **RGB + Artifact Map Ensemble** 방식을 적용.

---

## 1. 프로젝트 개요

딥페이크 기술의 품질이 높아지면서 가짜 뉴스, 얼굴 도용, 명예훼손, 사생활 침해와 같은 문제가 증가하고 있다.
특히 최근 생성·편집된 얼굴 이미지는 사람 눈으로 Real/Fake를 구분하기 어려운 경우가 많아 이미지 기반 자동 탐지 모델의 필요성이 커진다.

본 프로젝트의 목표는 다음과 같음.

* Real 얼굴 이미지와 AI 기반 얼굴 조작 이미지(Fake) 구분
* RGB 이미지와 Artifact Map을 함께 활용해 탐지 성능 개선
* CNN 기반 모델과 Transformer 기반 모델을 비교하여 최종 모델 선정


---

## 2. 데이터셋 구성

본 프로젝트에서는 Real 이미지 500장, Fake 이미지 500장으로 총 1000장의 이미지 데이터셋을 구성.

| Class | Count |
| ----- | ----: |
| Real  |   500 |
| Fake  |   500 |
| Total |  1000 |


Train과 Validation은 8:2 비율로 분할.

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
실행 전 위 폴더 구조에 맞게 데이터를 배치.

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

Artifact Map은 사람이 직접 Fake 여부를 판별하기 위한 이미지가 아니라, CNN 모델이 미세한 질감, 경계선, 노이즈 패턴을 보조적으로 학습하도록 만든 입력한다.



---

### 3.3 RGB + Artifact Map Ensemble

RGB 모델과 Artifact Map 모델은 각각 따로 학습한다.

이후 두 모델의 Fake probability를 평균하여 최종 Fake probability를 계산한다.

```text
Final Fake Probability = (RGB Fake Probability + Artifact Fake Probability) / 2
```

최종 판정 기준은 threshold 0.5.

```text
Fake Probability >= 0.5 → FAKE
Fake Probability < 0.5  → REAL
```

앙상블을 사용한 이유는 RGB 이미지와 Artifact Map이 서로 다른 정보를 담고 있기 때문이다.

* RGB Model: 얼굴의 전체 구조, 색상, 조명 정보 학습
* Artifact Model: 피부 질감, 얼굴 경계선, 노이즈와 같은 고주파 성분 학습

따라서 두 모델의 결과를 결합하면 단일 입력 모델보다 더 안정적인 판단 가능함.

<img width="584" height="468" alt="스크린샷_18-6-2026_162831_" src="https://github.com/user-attachments/assets/75ae25f6-022a-4ebf-9499-0d88cf38094c" />

---

## 4. 모델 선정 과정

본 프로젝트에서는 최종 backbone을 선정하기 위해 먼저 EfficientNet 계열 모델을 비교함.
EfficientNet은 B0부터 B4까지 모델 크기와 입력 해상도가 단계적으로 확장되는 구조이기 때문에, 딥페이크 탐지에 적합한 모델 규모를 확인하기 위해 후보 모델로 사용함.

### 4.1 EfficientNet 계열 1차 실험

먼저 EfficientNet-B0부터 EfficientNet-B4까지 동일한 입력 해상도인 224×224 조건에서 성능을 비교함.
이 실험은 모델 크기가 커질수록 성능이 항상 좋아지는지 확인하기 위한 1차 비교 실험임.

| Model           | Input Size | Purpose                        |
| --------------- | ---------: | ------------------------------ |
| EfficientNet-B0 |    224×224 | 가장 작은 EfficientNet 모델 기준 성능 확인 |
| EfficientNet-B1 |    224×224 | B0보다 확장된 모델 성능 확인              |
| EfficientNet-B2 |    224×224 | 중간 규모 모델 성능 확인                 |
| EfficientNet-B3 |    224×224 | 더 큰 EfficientNet 계열 성능 확인      |
| EfficientNet-B4 |    224×224 | 최종 후보 모델의 동일 해상도 조건 성능 확인      |

실험 결과, EfficientNet 계열에서 모델 크기가 커진다고 해서 224×224 입력 조건에서 성능이 항상 향상되지는 않음을 확인함.
이를 통해 단순히 모델 크기만 비교하는 것이 아니라, 각 모델에 적합한 입력 해상도도 함께 고려해야 한다고 판단함.

| Model | Size | Accuracy |
|---|---:|---:|
| EfficientNet-B0 | 224×224 | 76.55% |
| EfficientNet-B1 | 224×224 | 75.68% |
| EfficientNet-B2 | 224×224 | 73.80% |
| EfficientNet-B3 | 224×224 | 71.24% |
| EfficientNet-B4 | 224×224 | 67.70% |
| EfficientNet-B5 | 224×224 | 57.70% |


---

### 4.2 권장 입력 해상도 적용 실험

EfficientNet은 모델별 권장 입력 해상도가 존재함.
특히 EfficientNet-B4는 380×380 입력 해상도를 권장하기 때문에, B4 모델에 380×380 해상도를 적용하여 추가 실험을 진행.

권장 해상도 적용 후 EfficientNet-B4의 성능이 향상되는 것을 확인함.
따라서 본 프로젝트에서는 얼굴의 미세한 질감, 경계선, 노이즈와 같은 정보를 더 잘 반영할 수 있는 EfficientNet-B4를 최종 backbone으로 선정함.

| Model | Size | Accuracy |
|---|---:|---:|
| EfficientNet-B4 | 224×224 | 67.70% |
| EfficientNet-B4 | 380×380 | 85.90% |
| EfficientNet-B5 | 224×224 | 57.70% |
| EfficientNet-B5 | 456×456 | 71.40% |
```

---

### 4.3 최종 비교 모델 선정

EfficientNet-B4를 최종 후보 모델로 선정한 뒤, 다른 구조의 모델들과 성능을 비교하기 위해 총 4개의 모델을 실험 대상으로 선정함.

| Model             | 선정 이유                                     |
| ----------------- | ----------------------------------------- |
| EfficientNet-B4   | 권장 해상도 적용 후 성능이 향상되어 최종 backbone 후보로 선정   |
| MobileNetV3-Large | 경량 CNN 모델로, 실제 서비스 적용 가능성 비교를 위해 선정       |
| ResNet-50         | 이미지 분류에서 널리 사용되는 전통적인 CNN baseline으로 선정   |
| ViT-B/16          | CNN과 다른 Transformer 기반 이미지 모델과의 비교를 위해 선정 |

최종적으로 EfficientNet-B4는 RGB 입력과 Artifact Map 입력 모두에서 안정적인 성능을 보였고, RGB + Artifact Map 앙상블에서도 가장 높은 정확도를 기록함.
따라서 본 프로젝트의 최종 모델은 **EfficientNet-B4 기반 RGB + Artifact Map Ensemble 구조**로 선정함.

MobileNetV3-Large는 경량 모델로서 높은 성능을 보였으나, 본 프로젝트에서는 속도보다 최종 정확도와 입력별 안정성을 더 중요하게 판단함.
ResNet-50과 ViT-B/16은 이번 데이터셋 및 학습 조건에서는 성능이 낮게 나타남. 특히 ViT-B/16은 모든 이미지를 Fake로 예측하는 클래스 쏠림 현상이 발생하여 추가 검증이 필요한 결과로 해석함.


## 5. 학습 설정

| Parameter            | Value               |
| -------------------- | ------------------- |
| Backbone             | EfficientNet-B4     |
| Input Size           | 380 × 380           |
| Batch Size           | 8                   |
| Epochs               | 10                  |
| Early Stopping       | patience 4          |
| Checkpoint Criterion | Min Validation Loss |

Min Validation Loss는 validation loss가 가장 낮게 나온 시점의 모델을 checkpoint로 저장한다는 의미임.
즉, 마지막 epoch의 모델을 사용하는 것이 아니라 검증 데이터에서 오차가 가장 작았던 모델을 최종 평가에 사용함.

---

## 6. 실행 방법


### 6.1 RGB 모델 학습

본 프로젝트는 하나의 `config.yaml` 파일에서 `input_mode` 값을 변경하여 RGB 모델과 Artifact Map 모델을 각각 학습함.

RGB 모델을 학습할 때는 `config.yaml`을 다음과 같이 설정함.

```yaml
input_mode: "rgb"
checkpoint_filename: "rgb_model"
```

이후 학습 실행.

```bash
python main_trainer.py
```

학습 완료 후 다음 checkpoint 저장한.

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

이후 학습 실행.

```bash
python main_trainer.py
```

학습 완료 후 다음 checkpoint 저장됨.

```text
models/artifact_map_model.ckpt
```

주의: `classify.py`의 기본 설정은 `models/rgb_model.ckpt`와 `models/artifact_map_model.ckpt`를 불러오도록 되어 있음.
따라서 checkpoint 이름을 위와 같이 맞추는 것을 권장함.

---

### 6.3 단일 이미지 예측

학습된 RGB 모델과 Artifact Map 모델을 이용하여 단일 이미지에 대해 Real/Fake 예측 가능함.

```bash
python classify.py path/to/image.jpg
```

기본 checkpoint 경로는 다음과 같음.

```text
models/rgb_model.ckpt
models/artifact_map_model.ckpt
```

다른 checkpoint를 사용할 경우 다음과 같이 지정 가능함.

```bash
python classify.py path/to/image.jpg --rgb_model models/rgb_model.ckpt --artifact_model models/artifact_map_model.ckpt
```

출력 예시는 다음과 같음.

```text
Prediction: FAKE
[Ensemble Result]
Real: 0.338 | Fake: 0.662
```

Fake probability가 0.5 이상이면 Fake로 분류됨.

**[이미지 삽입 위치]**
PPT의 `모델 학습 후 실행` 화면 또는 Real/Fake 예측 결과 이미지 삽입 권장.

```markdown
![Prediction Example](assets/prediction_example.png)
```

---

## 7. 실험 결과

### 7.1 RGB / Artifact 단일 입력 성능 비교

RGB 입력과 Artifact Map 입력을 각각 따로 사용하여 모델별 성능 비교함.

**[표 삽입 위치]**
PPT의 `RGB / Artifact 단일 입력 성능 비교 결과 표` 삽입 권장.

| Model             | RGB Accuracy | Artifact Accuracy |
| ----------------- | -----------: | ----------------: |
| EfficientNet-B4   |       90.00% |            91.50% |
| MobileNetV3-Large |       91.50% |            87.00% |
| ResNet-50         |       52.00% |            54.00% |
| ViT-B/16          |       50.00% |            50.00% |


<img width="1320" height="888" alt="image" src="https://github.com/user-attachments/assets/06124e41-dcd7-4264-a7f8-ae63f042e151" />


결과적으로 RGB 입력에서는 MobileNetV3와 EfficientNet-B4가 높은 성능을 보였고, Artifact Map 입력에서는 EfficientNet-B4가 가장 안정적인 성능을 보임.

---

### 7.2 RGB + Artifact Map Ensemble 성능 비교

RGB 모델과 Artifact Map 모델의 Fake probability를 평균하여 최종 예측 수행함.

**[표 삽입 위치]**
PPT의 `RGB & Artifact 결합 입력 기반 모델 성능 비교 결과 표` 삽입 권장.

| Model             | Ensemble Accuracy |
| ----------------- | ----------------: |
| EfficientNet-B4   |            93.50% |
| MobileNetV3-Large |            92.50% |
| ResNet-50         |            55.50% |
| ViT-B/16          |            50.00% |

**[그래프 삽입 위치]**
PPT의 `RGB + Artifact Map Ensemble 모델별 성능 비교 그래프` 삽입 권장.

```markdown
![Ensemble Accuracy Result](assets/ensemble_accuracy_result.png)
```

EfficientNet-B4는 최종 앙상블 정확도 93.50%로 가장 높은 성능을 보임.
따라서 최종 모델은 EfficientNet-B4 기반 RGB + Artifact Map Ensemble 구조로 선정함.

---

### 7.3 Threshold 및 Confusion Matrix 분석

최종 EfficientNet-B4 Ensemble 모델은 threshold 0.5 기준으로 평가함.

| Metric       |   Result |
| ------------ | -------: |
| Accuracy     |   93.50% |
| Precision    |    0.958 |
| Recall       |    0.910 |
| F1-score     |    0.933 |
| Real Correct | 96 / 100 |
| Fake Correct | 91 / 100 |

Confusion Matrix:

|             | Pred Real | Pred Fake |
| ----------- | --------: | --------: |
| Actual Real |        96 |         4 |
| Actual Fake |         9 |        91 |

**[이미지 삽입 위치]**
PPT의 `Threshold 및 Confusion Matrix 분석` 이미지 삽입 권장.

```markdown
![Threshold and Confusion Matrix](assets/confusion_matrix_threshold.png)
```

---

## 8. 결과 해석

EfficientNet-B4는 RGB 입력과 Artifact Map 입력 모두에서 안정적인 성능을 보였고, 최종 앙상블에서도 가장 높은 정확도 기록함.

MobileNetV3-Large는 경량 모델임에도 높은 성능을 보였으나, Artifact Map 입력에서 EfficientNet-B4보다 낮은 성능을 보임.
따라서 본 프로젝트에서는 속도보다 탐지 성능과 입력별 안정성을 우선하여 EfficientNet-B4를 최종 모델로 선정함.

ResNet-50은 전통적인 CNN baseline으로 사용했지만, 이번 데이터셋 조건에서는 성능이 낮게 나타남.

ViT-B/16은 모든 이미지를 Fake로 예측하는 클래스 쏠림 현상이 발생함.
이는 데이터셋 규모, 입력 해상도, fine-tuning 설정 차이의 영향을 받은 것으로 해석했으며, 추가 검증이 필요한 결과로 판단함.

**[이미지 삽입 위치]**
PPT의 `모델 비교 결과 및 해석` 슬라이드 이미지 또는 모델별 해석 요약 그래프 삽입 가능함.

```markdown
![Model Comparison Interpretation](assets/model_comparison_interpretation.png)
```

---

## 9. 한계점

### 9.1 데이터셋 유형

본 프로젝트의 Fake 데이터는 AI 기반 표정 조작 및 얼굴 편집 이미지 중심으로 구성됨.
따라서 모든 유형의 딥페이크 일반 탐지보다는 face manipulation 이미지 탐지로 해석할 필요 있음.

### 9.2 Artifact Map의 한계

Artifact Map은 고주파 성분을 강조하지만, 조명, 화장, 피부 질감, 압축 노이즈도 함께 반영될 수 있음.

### 9.3 데이터 편향 가능성

로고, 워터마크, 배경 차이가 특정 클래스에만 존재할 경우 모델이 얼굴 조작 특징이 아닌 외부 요소에 영향을 받을 수 있음.

---

## 10. 향후 과제

### 10.1 데이터 다양화

Face Swap, 표정 조작, GAN 생성 등 다양한 얼굴 조작 유형을 추가하여 모델의 일반화 성능 향상 가능함.

### 10.2 전처리 고도화

얼굴 crop 및 alignment를 적용하여 얼굴 영역 중심의 탐지 성능 개선 가능함.

### 10.3 분석 및 서비스 확장

주파수 영역 분석, 영상 기반 temporal artifact 분석, 웹 서비스 연동 등으로 확장 가능함.

---

## 11. Repository Structure

```text
deepfake-detection/
├── main_trainer.py
├── evaluate.py
├── classify.py
├── config.yaml
├── datasets/
│   └── hybrid_loader.py
├── lightning_modules/
│   └── detector.py
├── models/
│   └── .gitkeep
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── validation/
│       ├── real/
│       └── fake/
├── assets/
│   ├── artifact_map_generation.png
│   ├── proposed_ensemble_architecture.png
│   ├── rgb_artifact_single_input_result.png
│   ├── ensemble_accuracy_result.png
│   ├── confusion_matrix_threshold.png
│   └── prediction_example.png
└── README.md
```

---

## 12. Notes

* 이미지 데이터셋과 학습된 checkpoint 파일은 용량 문제로 GitHub에 포함하지 않음.
* RGB 모델과 Artifact Map 모델은 동일한 학습 코드에서 `config.yaml`의 `input_mode` 값만 변경하여 각각 학습함.
* 최종 예측은 RGB 모델과 Artifact Map 모델의 Fake probability를 평균하여 계산함.
* `models/rgb_model.ckpt`와 `models/artifact_map_model.ckpt`가 있어야 `classify.py`를 통한 앙상블 예측 가능함.
