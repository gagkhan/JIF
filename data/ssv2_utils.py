import os


def clean_empty_folders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if not os.listdir(folder_path):
                print(f"Removing empty folder: {folder_path}")
                os.rmdir(folder_path)


# Specify the directory you want to clean up
smth_smth_root = "data/20bn-something-something-v2-frames"


import numpy as np


def create_ssv2_tiny(num_images=32768 * 8):

    # The number of videos in the something-something-v2 dataset is 214671.
    MAX_NUM_VIDEOS = 214671
    # Not all videos have gotten frames extracted.

    np.random.seed(0)

    selected_video_indices = []
    k = 0
    while k < num_images:
        video_index = np.random.choice(MAX_NUM_VIDEOS, replace=False)
        # Check if the sampled video indices are valid

        if video_index not in selected_video_indices and os.path.exists(
            f"{smth_smth_root}/{video_index}"
        ):

            selected_video_indices.append(video_index)

            # check the number of frames in the video directory
            num_frames = len(os.listdir(f"{smth_smth_root}/{video_index}"))

            # create a new directory for the tiny dataset
            os.makedirs("data/20bn-something-something-v2-frames-small", exist_ok=True)

            # copy directory to new location
            os.system(
                f"cp -r {smth_smth_root}/{video_index} data/20bn-something-something-v2-frames-small/"
            )

            k += num_frames
            print(f"Number of images: {k}")

    print(f"Number of videos: {len(selected_video_indices)}")


if __name__ == "__main__":
    # clean_empty_folders(directory_to_clean)

    create_ssv2_tiny()
