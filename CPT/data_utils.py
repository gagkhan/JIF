import os

import data_utils
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video


def get_ssv2_frames_root(scale="tiny"):

    data_root = os.path.join(os.environ["DATA_ROOT"], f"20bn-something-something-v2-frames-{scale}")

    return data_root


def get_ours_root():
    data_root = os.path.join(os.environ["DATA_ROOT"], f"ours_v0_frames")
    return data_root


class SSV2Dataset(Dataset):
    def __init__(self, data_root, transform, skip_frames=5):

        self.data_root = data_root
        self.transform = transform
        self.skip_frames = skip_frames  # k, gap between o_t and o_t+k+1

        # Print dataset root
        # print("Dataset root:", self.data_root)

        # print("Processing dataset...")

        # Count the number of frames in each demo
        # Go through each folder and count the number of frames
        self.frames_per_demo = []
        self.path_to_frames = []
        self.path_to_folders = []
        for folder in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder)
            self.path_to_folders.append(folder_path)
            frames = sorted(os.listdir(folder_path))
            self.path_to_frames.append(frames)
            num_frames = len(frames)
            self.frames_per_demo.append(num_frames)

        # print("Frame paths:", self.path_to_frames[0])

        # print("Number of demos:", len(self.frames_per_demo))
        # print(
        #     "Avg. number of frames per demo:",
        #     int(sum(self.frames_per_demo) / len(self.frames_per_demo)),
        # )

        # Compute the length of the dataset
        # Number of o_t, o_t+k+1, o_g tuples in the dataset
        self.ntuples_per_demo = []
        # print("Computing length of dataset...")
        length = 0
        self.index_to_demo_index = {}
        for i, frames in enumerate(self.frames_per_demo):
            # formula: demo_length = frames - seq_len + 1
            demo_length = frames - self.skip_frames - 1
            for j in range(demo_length):
                self.index_to_demo_index[length + j] = (i, j)
            length += demo_length
            self.ntuples_per_demo.append(demo_length)

        # print("Number of tuples per demo:", self.ntuples_per_demo)
        # print("Length of dataset:", length)
        self.cumsum_ntuples_per_demo = np.cumsum(self.ntuples_per_demo)

        # print("Cumulative sum of tuples per demo:", self.cumsum_ntuples_per_demo)

    def __len__(self):
        return self.cumsum_ntuples_per_demo[-1]

    def __getitem__(self, index):
        i, j = self.index_to_demo_index[index]
        current_frame = os.path.join(self.path_to_folders[i], self.path_to_frames[i][j])
        next_frame = os.path.join(
            self.path_to_folders[i], self.path_to_frames[i][j + self.skip_frames + 1]
        )
        goal_frame = os.path.join(self.path_to_folders[i], self.path_to_frames[i][-1])

        # print("Current frame:", current_frame)
        # print("Next frame:", next_frame)
        # print("Goal frame:", goal_frame)

        # Load images with PIL
        current_image = Image.open(os.path.join(self.data_root, current_frame))
        next_image = Image.open(os.path.join(self.data_root, next_frame))
        goal_image = Image.open(os.path.join(self.data_root, goal_frame))

        # Apply transformations
        current_image = self.transform(current_image)
        next_image = self.transform(next_image)
        goal_image = self.transform(goal_image)

        return current_image, next_image, goal_image


def test_ssv2_dataset():
    data_root = data_utils.get_ssv2_frames_root(scale="tiny")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = SSV2Dataset(data_root, transform)
    print(len(dataset))
    print([image.shape for image in dataset[0]])


def test_our_dataset():
    data_root = data_utils.get_ours_root()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = SSV2Dataset(data_root, transform)
    print(len(dataset))
    print([image.shape for image in dataset[0]])


class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Get a list of video files or frame directories
        self.video_files = [...]  # List of video file paths or frame directories

        self.video_files = [
            os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".mp4")
        ]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]

        # Read the video frames
        frames, audio, info = read_video(video_path)

        # Assuming frames is a tensor of shape (T, H, W, C), where T is the number of frames
        # You can modify this part based on the actual structure of your data

        # Extract current, next, and goal frames
        sub_idx = torch.randint(high=frames.shape[0] - 2, size=(1,)).item()
        current_frame = frames[sub_idx]  # All frames except the last two
        next_frame = frames[sub_idx + 1]  # All frames except the first and last
        goal_frame = frames[-1]  # All frames except the first two

        # Apply transformations if provided
        if self.transform:
            current_frame = self.transform(current_frame)
            next_frame = self.transform(next_frame)
            goal_frame = self.transform(goal_frame)

        # Convert to torch tensors
        # current_frame = torch.from_numpy(current_frame)
        # next_frame = torch.from_numpy(next_frame)
        # goal_frame = torch.from_numpy(goal_frame)

        return current_frame, next_frame, goal_frame


def test_video_dataset():
    data_dir = os.path.join(os.environ["DATA_ROOT"], "ours_v0")
    transform = transforms.Compose(
        [transforms.Resize((256, 256))]
    )  # You can add more transformations

    video_dataset = VideoDataset(data_dir, transform=transform)

    # Access a sample from the dataset
    sample = video_dataset[0]
    current_frame, next_frame, goal_frame = sample

    # Print shapes of frames
    print("Current Frame Shape:", current_frame.shape)
    print("Next Frame Shape:", next_frame.shape)
    print("Goal Frame Shape:", goal_frame.shape)


if __name__ == "__main__":
    # test_ssv2_dataset()
    # test_video_dataset()

    test_our_dataset()
