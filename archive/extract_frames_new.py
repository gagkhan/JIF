import os

import cv2


def extract_frames_from_video(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_FPS, 10)

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Extracting frames from '{video_name}' with {fps} FPS and {total_frames} frames...")

    # Read and save each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Save the frame as an image
        frame_filename = f"{video_name}_frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    print(f"Frames extraction complete. {frame_count} frames saved in '{output_dir}'.")

    # Release the video capture object
    cap.release()


def main():
    data_dir = "path/to/your/data"

    # Get a list of video files
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mp4")]

    for video_file in video_files:
        output_dir = os.path.splitext(video_file)[0]
        extract_frames_from_video(video_file, output_dir)


def test():
    extract_frames_from_video(
        "/home/gagan/Home/VideoIL/data/ours_v0/twohand_closemarble.mp4", "test_extract_frames"
    )


if __name__ == "__main__":
    # main()
    test()
