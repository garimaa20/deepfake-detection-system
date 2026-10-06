import cv2
import mediapipe as mp
from pathlib import Path
from model import predict

# -----------------------------
# File paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "test_images" / "test1.jpg"
cropped_path = BASE_DIR / "test_images" / "cropped_face.jpg"
detector_path = BASE_DIR / "face_detector.task"

# -----------------------------
# Check files
# -----------------------------
if not detector_path.exists():
    raise FileNotFoundError(
        f"Could not find {detector_path}\n"
        "Make sure face_detector.task is in the project folder."
    )

image = cv2.imread(str(input_path))

if image is None:
    raise FileNotFoundError(f"Could not open image: {input_path}")

print("Original image loaded successfully.")

# -----------------------------
# MediaPipe Face Detector
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=str(detector_path)
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.5
)

# Convert BGR -> RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

# -----------------------------
# Detect face
# -----------------------------
with FaceDetector.create_from_options(options) as detector:

    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        print("No face detected.")
        print("Running prediction on the original image instead.")
        result = predict(str(input_path))

    else:
        print("Face detected successfully.")

        detection = detection_result.detections[0]

        bbox = detection.bounding_box

        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = bbox.width
        h = bbox.height

        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        face = image[y:y2, x:x2]

        if face.size == 0:
            print("Face crop failed.")
            result = predict(str(input_path))
        else:
            cv2.imwrite(str(cropped_path), face)

            print("Face cropped successfully.")
            print("Cropped face saved to:", cropped_path)

            # Run deepfake model on cropped face
            result = predict(str(cropped_path))

# -----------------------------
# Final result
# -----------------------------
print()
print("==============================")
print("   DEEPFAKE DETECTION RESULT")
print("==============================")
print("Prediction :", result["prediction"])
print("Confidence :", result["confidence"], "%")
print("Fake Probability :", result["fake_probability"], "%")
print("Real Probability :", result["real_probability"], "%")
print("==============================")