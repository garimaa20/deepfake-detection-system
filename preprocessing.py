import cv2


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Image could not be loaded")
        return None

    # Resize
    image = cv2.resize(image, (224, 224))

    # BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    image = image / 255.0

    return image


# Test
processed_image = preprocess_image("test_images/test.jpg")

if processed_image is not None:
    print("Image preprocessing successful!")
    print("Final shape:", processed_image.shape)
    print("Pixel range:", processed_image.min(), "to", processed_image.max())