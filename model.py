import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

print("Loading deepfake detection model...")

# Load processor
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

# Load model
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)

# CPU inference
model.eval()

# IMPORTANT:
# This model uses:
# 0 = Fake
# 1 = Real
ID2LABEL = {
    0: "Fake",
    1: "Real"
}

print("Model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Preprocess image
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    # Find highest probability
    predicted_class = torch.argmax(probabilities).item()

    # Get label
    predicted_label = ID2LABEL[predicted_class]

    # Confidence
    confidence = probabilities[predicted_class].item() * 100

    # Get individual probabilities
    fake_probability = probabilities[0].item() * 100
    real_probability = probabilities[1].item() * 100

    # Print debugging information
    print()
    print("Raw model labels:", ID2LABEL)
    print("Raw logits:", outputs.logits)
    print("Fake probability:", round(fake_probability, 2), "%")
    print("Real probability:", round(real_probability, 2), "%")
    print("Predicted class:", predicted_class)
    print("Predicted label:", predicted_label)

    # Return result for Streamlit/integration
    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_probability, 2),
        "real_probability": round(real_probability, 2)
    }


