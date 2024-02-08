import os

import data_utils
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_ssv2_frames_root(scale="tiny"):

    data_root = os.path.join(os.environ["DATA_ROOT"], f"20bn-something-something-v2-frames-{scale}")

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


def test_demo_dataset():
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


if __name__ == "__main__":
    test_demo_dataset()
