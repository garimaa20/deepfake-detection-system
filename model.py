import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

print("Loading deepfake detection model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded successfully!")


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits, dim=1
    )[0]

    predicted_class = torch.argmax(probabilities).item()

    # Model mapping:
    # 0 = REAL
    # 1 = FAKE
    prediction = model.config.id2label[predicted_class]

    real_probability = probabilities[0].item() * 100
    fake_probability = probabilities[1].item() * 100

    return {
        "prediction": prediction,
        "confidence": round(
            probabilities[predicted_class].item() * 100, 2
        ),
        "real_probability": round(real_probability, 2),
        "fake_probability": round(fake_probability, 2)
    }