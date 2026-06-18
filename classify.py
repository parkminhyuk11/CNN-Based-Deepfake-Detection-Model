import torch
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image, ImageFilter
import numpy as np
import argparse



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



def load_model(model_path):
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    model = efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("model.", "").replace("backbone.", "")
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict)
    model.eval()

    return model



rgb_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])



artifact_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    ArtifactMapTransform(blur_radius=2),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict_probs(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    return real_prob, fake_prob



def predict_image_ensemble(image_path, rgb_model, artifact_model):
    rgb_real, rgb_fake = predict_probs(
        image_path,
        rgb_model,
        rgb_transform
    )

    artifact_real, artifact_fake = predict_probs(
        image_path,
        artifact_model,
        artifact_transform
    )

    # 기본 앙상블 방식: RGB와 Artifact를 동일 비율로 반영
    ensemble_fake = (rgb_fake + artifact_fake) / 2
    ensemble_real = 1 - ensemble_fake

    pred = 1 if ensemble_fake >= 0.5 else 0
    label = "FAKE" if pred == 1 else "REAL"

    print("\n=============================================")
    print(f"Prediction: {label}")
    print("---------------------------------------------")
    print("[Ensemble Result]")
    print(f"Real: {ensemble_real:.3f} | Fake: {ensemble_fake:.3f}")
    print("=============================================")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path",
        help="Path to image file (.jpg/.png)"
    )

    parser.add_argument(
        "--rgb_model",
        default="models/rgb_model.ckpt",
        help="Path to RGB EfficientNet-B4 model checkpoint"
    )

    parser.add_argument(
        "--artifact_model",
        default="models/artifact_map_model.ckpt",
        help="Path to Artifact EfficientNet-B4 model checkpoint"
    )

    args = parser.parse_args()

    rgb_model = load_model(args.rgb_model)
    artifact_model = load_model(args.artifact_model)

    predict_image_ensemble(
        args.image_path,
        rgb_model,
        artifact_model
    )