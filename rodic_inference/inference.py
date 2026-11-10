from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms


INFERENCE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = INFERENCE_DIR.parent
SRC_DIR = DEPLOY_DIR / "src"
MODEL_DIR = DEPLOY_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.model import build_model


IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.50

CODEBRIM_CLASSES = [
    "background",
    "crack",
    "spallation",
    "efflorescence",
    "exposed_bars",
    "corrosion_stain",
]


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def ingest_image_bytes(image_bytes):
    if image_bytes is None:
        raise ValueError("No image data supplied.")

    if len(image_bytes) == 0:
        raise ValueError("Image data is empty.")

    maximum_bytes = 10 * 1024 * 1024

    if len(image_bytes) > maximum_bytes:
        raise ValueError(
            "Image exceeds the 10 MB upload limit."
        )

    buffer = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            "OpenCV could not decode the uploaded image."
        )

    height, width = image_bgr.shape[:2]

    if width < 32 or height < 32:
        raise ValueError(
            "Image dimensions are too small."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return image_rgb


def prepare_tensor(image_rgb):
    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    tensor = TRANSFORM(resized)

    return tensor.unsqueeze(0)


def load_checkpoint(model, checkpoint_path):
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model


def load_codebrim_model():
    model = build_model("codebrim")

    checkpoint = MODEL_DIR / "best_codebrim.pt"

    return load_checkpoint(
        model,
        checkpoint,
    )


def load_sdnet_model():
    model = build_model("sdnet")

    checkpoint = MODEL_DIR / "best_sdnet.pt"

    return load_checkpoint(
        model,
        checkpoint,
    )


def run_codebrim_inference(model, image_rgb):
    tensor = prepare_tensor(image_rgb)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0]

    probabilities = probabilities.cpu().numpy()

    results = []

    for class_name, probability in zip(
        CODEBRIM_CLASSES,
        probabilities,
    ):
        score = float(probability)

        results.append({
            "defect": class_name,
            "score": score,
            "detected": score >= CONFIDENCE_THRESHOLD,
        })

    return results


def run_sdnet_inference(model, image_rgb):
    tensor = prepare_tensor(image_rgb)

    with torch.inference_mode():
        logits = model(tensor)

    probability = float(
        torch.sigmoid(logits).reshape(-1)[0].item()
    )

    return {
        "crack_probability": probability,
        "crack_detected": probability >= CONFIDENCE_THRESHOLD,
    }
