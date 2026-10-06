import cv2
import os


def extract_frames(video_path, output_folder, num_frames=10):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Video could not be opened")
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total frames:", total_frames)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Calculate interval between frames
    interval = max(total_frames // num_frames, 1)

    count = 0
    saved = 0

    while saved < num_frames:

        success, frame = video.read()

        if not success:
            break

        if count % interval == 0:

            filename = os.path.join(
                output_folder,
                f"frame_{saved + 1}.jpg"
            )

            cv2.imwrite(filename, frame)

            saved += 1

        count += 1

    video.release()

    print("Frames extracted:", saved)


# Test
extract_frames(
    "test_videos/test.mp4",
    "test_videos/frames",
    10
)