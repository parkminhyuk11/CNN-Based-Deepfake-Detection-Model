import os
import torch
import numpy as np

from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class ArtifactMapTransform:
    def __init__(self, blur_radius=2):
        self.blur_radius = blur_radius

    def __call__(self, img):
        img = img.convert("RGB")

        blurred = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        img_np = np.array(img).astype(np.float32)
        blurred_np = np.array(blurred).astype(np.float32)

        artifact = np.abs(img_np - blurred_np)
        artifact = artifact / (artifact.max() + 1e-8) * 255
        artifact = artifact.astype(np.uint8)

        return Image.fromarray(artifact)



def build_model():
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features

    model.classifier[2] = torch.nn.Dropout(p=0.4)
    model.classifier[3] = torch.nn.Linear(in_features, 2)

    return model



def load_model(model_path, device):
    print(f"모델 불러오기: {model_path}")

    model = build_model()

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    clean_state_dict = {}

    for k, v in state_dict.items():
        clean_key = k.replace("model.", "")
        clean_key = clean_key.replace("backbone.", "")
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    return model




rgb_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

artifact_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    ArtifactMapTransform(blur_radius=2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_fake_prob(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    fake_prob = probs[1].item()
    return fake_prob



def evaluate_folder(folder_path, correct_label, rgb_model, artifact_model, device):
    rgb_correct = 0
    artifact_correct = 0
    ensemble_correct = 0
    total = 0

    if not os.path.exists(folder_path):
        print(f"폴더를 찾을 수 없습니다: {folder_path}")
        return 0, 0, 0, 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)

            try:
                rgb_fake = predict_fake_prob(
                    img_path,
                    rgb_model,
                    rgb_transform,
                    device
                )

                artifact_fake = predict_fake_prob(
                    img_path,
                    artifact_model,
                    artifact_transform,
                    device
                )

                rgb_pred = 1 if rgb_fake >= 0.5 else 0
                artifact_pred = 1 if artifact_fake >= 0.5 else 0

                ensemble_fake = (rgb_fake + artifact_fake) / 2
                ensemble_pred = 1 if ensemble_fake >= 0.5 else 0

                if rgb_pred == correct_label:
                    rgb_correct += 1

                if artifact_pred == correct_label:
                    artifact_correct += 1

                if ensemble_pred == correct_label:
                    ensemble_correct += 1

                total += 1

            except Exception as e:
                print(f"이미지 처리 실패: {img_path} / 오류: {e}")

    return rgb_correct, artifact_correct, ensemble_correct, total



if __name__ == "__main__":
    print("=" * 60)
    print("MobileNetV3-Large RGB + Artifact Map Ensemble 평가")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")


    rgb_model_path = "models/mobilenetv3_rgb_model.ckpt"
    artifact_model_path = "models/mobilenetv3_artifact_model.ckpt"


    real_folder = "data/validation/real"
    fake_folder = "data/validation/fake"

    rgb_model = load_model(rgb_model_path, device)
    artifact_model = load_model(artifact_model_path, device)

    real_rgb, real_art, real_ens, real_total = evaluate_folder(
        real_folder,
        correct_label=0,
        rgb_model=rgb_model,
        artifact_model=artifact_model,
        device=device
    )

    fake_rgb, fake_art, fake_ens, fake_total = evaluate_folder(
        fake_folder,
        correct_label=1,
        rgb_model=rgb_model,
        artifact_model=artifact_model,
        device=device
    )

    total = real_total + fake_total

    if total == 0:
        print("평가할 이미지가 없습니다. validation 경로를 확인하세요.")
        exit()

    rgb_correct = real_rgb + fake_rgb
    artifact_correct = real_art + fake_art
    ensemble_correct = real_ens + fake_ens

    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)

    print(f"REAL 검증 이미지 수: {real_total}")
    print(f"FAKE 검증 이미지 수: {fake_total}")
    print(f"총 검증 이미지 수: {total}")

    print("-" * 60)

    print("[RGB 모델]")
    print(f"REAL 정답: {real_rgb}/{real_total}")
    print(f"FAKE 정답: {fake_rgb}/{fake_total}")
    print(f"전체 정답: {rgb_correct}/{total}")
    print(f"Accuracy: {rgb_correct / total * 100:.2f}%")

    print("-" * 60)

    print("[Artifact Map 모델]")
    print(f"REAL 정답: {real_art}/{real_total}")
    print(f"FAKE 정답: {fake_art}/{fake_total}")
    print(f"전체 정답: {artifact_correct}/{total}")
    print(f"Accuracy: {artifact_correct / total * 100:.2f}%")

    print("-" * 60)

    print("[RGB + Artifact Map Ensemble]")
    print(f"REAL 정답: {real_ens}/{real_total}")
    print(f"FAKE 정답: {fake_ens}/{fake_total}")
    print(f"전체 정답: {ensemble_correct}/{total}")
    print(f"Accuracy: {ensemble_correct / total * 100:.2f}%")

    print("=" * 60)