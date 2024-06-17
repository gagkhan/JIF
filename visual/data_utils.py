import os
import pickle

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VisDemoBase(Dataset):

    def __init__(self, data_root, transform, skip_frames=5, action_only=False):

        self.data_root = data_root
        self.transform = transform
        assert self.transform is not None, "None transform is not supported"
        self.skip_frames = skip_frames  # k, gap between o_t and o_t+k+1
        self.action_only = action_only

        assert os.path.exists(data_root), "specified data_root does not exist"

        # Print dataset root
        # print("Dataset root:", self.data_root)

        # print("Processing dataset...")

        # Count the number of frames in each demo
        # Go through each folder and count the number of frames
        self.path_to_folders = []
        self.path_to_frames = []
        self.frames_per_demo = []
        for folder in os.listdir(self.data_root):
            folder_path = os.path.join(self.data_root, folder)

            # skip if the folder is NOT a directory
            if not os.path.isdir(folder_path):
                continue

            # process contents of the folder
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

        # We need to know the shape of actions to create the correct tensors
        # Hence, we save the shapes in a dictionary for easy access and load it here
        self.action_dim = 0
        if os.path.exists(os.path.join(data_root, "shapes.yaml")):
            self.shapes_dict = yaml.load(
                open(os.path.join(data_root, "shapes.yaml"), "r"),
                Loader=yaml.FullLoader,
            )
            self.action_dim = self.shapes_dict["action_dim"]

    def _get_act_chunk(self, demo_idx, start_idx, chunk_size):
        actions = torch.zeros([chunk_size, self.action_dim], dtype=torch.float32)
        amask = torch.zeros_like(actions)
        action_path = os.path.join(self.path_to_folders[demo_idx], "actions.npy")
        if os.path.exists(action_path):
            actions[:] = torch.from_numpy(np.load(action_path))[start_idx : start_idx + chunk_size]
            amask = torch.ones_like(actions)

        return actions, amask

    def _get_img(self, demo_idx, frame_idx):
        path = os.path.join(self.data_root, self.path_to_folders[demo_idx], self.path_to_frames[demo_idx][frame_idx])
        return self.transform(Image.open(path))


class VisDemoDataset(VisDemoBase):

    def __init__(self, data_root, transform, skip_frames=5, action_only=False):

        super().__init__(data_root, transform, skip_frames, action_only)

        print("Frame paths:", self.path_to_frames[0])

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
        
        actions, amask = self._get_act_chunk(i, j, self.skip_frames + 1)

        if self.action_only:
            return actions, amask
        else:
            curr_img = self._get_img(i, j)
            next_img = self._get_img(i, j + self.skip_frames + 1)
            goal_img = self._get_img(i, -1)

            return curr_img, next_img, goal_img, actions, amask

    @property
    def action_shape(self):
        return (self.skip_frames + 1, self.action_dim)



class SeqVisDemoDataset(VisDemoBase):

    def __init__(
        self,
        data_root,
        transform=None,
        skip_frames=5,
        action_only=False,
        seq_len=5,
        ac_len=5,
    ):
        super().__init__(data_root, transform, skip_frames, action_only)
        self.seq_len = seq_len
        self.ac_len = ac_len

    def __len__(self):
        return len(self.path_to_folders)

    def __getitem__(self, index):
        index = None
        demo_idx = np.random.randint(0, len(self.path_to_folders))
        last_idx = np.random.randint(0, self.frames_per_demo[demo_idx])

        actions, amask = self._get_act_chunk(demo_idx, last_idx, self.ac_len)

        if self.action_only:
            return actions, amask
        else:
            idx = last_idx
            img_seq = []
            while len(img_seq) < self.seq_len:
                if idx > 0:
                    img = self._get_img(demo_idx, idx)
                    idx -= self.skip_frames
                else:
                    img = torch.zeros_like(img)
                img_seq.append(img)
            img_seq = torch.stack(img_seq)
            goal_img = self._get_img(demo_idx, -1)
            return img_seq, goal_img, actions, amask



def test_ssv2_tiny_dataset():

    scale = "tiny"
    data_root = os.path.join(os.environ["PROJDIR"], f"data/ssv2/20bn-something-something-v2-frames-{scale}")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = VisDemoDataset(data_root, transform)
    print(len(dataset))
    print([image.shape for image in dataset[0]])



def test_ours_v2_dataset():

    data_root = os.path.join(os.environ["PROJDIR"], f"data/ours/ours_v2_frames")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = VisDemoDataset(data_root, transform)
    print(len(dataset))
    print([image.shape for image in dataset[0]])


def test_seq_ours_v2_dataset():
    data_root = os.path.join(os.environ["PROJDIR"], f"data/ours/ours_v2_frames")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = SeqVisDemoDataset(data_root, transform, seq_len=2, ac_len=3)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        B = 4
        T = 2
        A = 3
        img_seq, goal, actions, amask = batch
        assert img_seq.shape[0] == B
        assert img_seq.shape[1] == T
        assert actions.shape[1] == A
        break


def test_nav2d():
    data_root = os.path.join(os.environ["PROJDIR"], f"data/nav2d/visual_v1")
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

    c, n, g, a, m = dataset[0]

    # print(m.shape)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
        break

    print(batch[4])


if __name__ == "__main__":

    # test_ssv2_tiny_dataset()
    # test_ours_v3_dataset()

    test_nav2d()
