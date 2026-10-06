import cv2
import os


def preprocess_frame(frame):
    # Resize
    frame = cv2.resize(frame, (224, 224))

    # BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Normalize
    frame = frame / 255.0

    return frame


def process_video(video_path, num_frames=10):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Video could not be opened")
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)

    processed_frames = []

    count = 0

    while len(processed_frames) < num_frames:

        success, frame = video.read()

        if not success:
            break

        if count % interval == 0:

            processed_frame = preprocess_frame(frame)

            processed_frames.append(processed_frame)

        count += 1

    video.release()

    return processed_frames


# Test
frames = process_video("test_videos/test.mp4", 10)

print("Processed frames:", len(frames))

if len(frames) > 0:
    print("Frame shape:", frames[0].shape)
    print("Pixel range:", frames[0].min(), "to", frames[0].max())