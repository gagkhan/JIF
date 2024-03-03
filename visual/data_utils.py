import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VisDemoDataset(Dataset):
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
            new_frames = []
            for frame in frames:
                if frame.endswith(".npy") or frame.endswith(".pkl"):
                    continue
                else:
                    new_frames.append(frame)
            self.path_to_frames.append(new_frames)
            num_frames = len(new_frames)
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

        actions = None
        action_file = os.path.join(self.path_to_folders[i], "actions.npy")
        if os.path.exists(action_file):
            actions = np.load(action_file)
            actions = actions[j : j + self.skip_frames + 1]

        # Apply transformations
        current_image = self.transform(current_image)[:3]
        next_image = self.transform(next_image)[:3]
        goal_image = self.transform(goal_image)[:3]
        if actions is not None:
            actions = torch.tensor(actions, dtype=torch.float32)
            return current_image, next_image, goal_image, actions
        else:
            return current_image, next_image, goal_image


def test_ssv2_tiny_dataset():

    scale = "tiny"
    data_root = os.path.join(
        os.environ["DATA_ROOT"], f"ssv2/20bn-something-something-v2-frames-{scale}"
    )
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = VisDemoDataset(data_root, transform)
    print(len(dataset))
    print([image.shape for image in dataset[0]])


def test_ours_v3_dataset():
    data_root = os.path.join(os.environ["DATA_ROOT"], f"ours/ours_v2_frames")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = VisDemoDataset(data_root, transform)
    print(len(dataset))
    print([image.shape for image in dataset[0]])


def test_nav2d():
    data_root = os.path.join(os.environ["DATA_ROOT"], f"nav2d_visual")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = VisDemoDataset(data_root, transform)

    print("Length of dataset:", len(dataset))

    dataset[0]
    # print([t for t in dataset[0]])

    c, n, g, a = dataset[0]

    print(a.shape)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        break


if __name__ == "__main__":

    # test_ssv2_tiny_dataset()
    # test_ours_v3_dataset()

    test_nav2d()
