import os
import pickle
import random
import re
from functools import partial
from typing import Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_img_dirs(data_root):
    """dirs containining .jpg are found"""
    image_extension = ".jpg"
    img_dirs = []
    for root, dirs, files in os.walk(data_root):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if any(file.lower().endswith(image_extension) for file in os.listdir(dir_path)):
                if not directory.startswith("depth"):  # ignores depth_images in bridge dataset
                    img_dirs.append(dir_path)
    return img_dirs


def load_dataset(
    args,
    wrapper_cls="VisDemoDataset",
    transform=None,
) -> Tuple[Dataset]:

    data_root = args.data_path
    train_split = args.train_split

    assert os.path.exists(data_root), "specified data_root does not exist"
    assert transform is not None, "None transform is not supported"
    img_dirs = get_img_dirs(data_root)
    random.shuffle(img_dirs)

    # print(f"Number of image directories: {len(img_dirs)}")

    train_dirs = img_dirs[: int(train_split * len(img_dirs))]
    val_dirs = img_dirs[int(train_split * len(img_dirs)) :]

    kwargs = dict()
    for param in ["skip_frames", "action_only", "use_ee", "seq_len", "action_chunk_len"]:
        if hasattr(args, param):
            kwargs[param] = args.__dict__[param]

    if wrapper_cls == "VisDemoDataset":
        dataset = partial(VisDemoDataset, data_root=data_root, transform=transform, **kwargs)
        train_dataset = dataset(demo_dirs=train_dirs)
        val_dataset = dataset(demo_dirs=val_dirs)
    elif wrapper_cls == "SeqVisDemoDataset":
        dataset = partial(SeqVisDemoDataset, data_root=data_root, transform=transform, **kwargs)
        train_dataset = dataset(demo_dirs=train_dirs)
        val_dataset = dataset(demo_dirs=val_dirs)

    return train_dataset, val_dataset


class VisDemoBase(Dataset):

    def __init__(self, data_root, demo_dirs, transform, skip_frames=5, action_only=False):

        self.data_root = data_root
        self.transform = transform
        # assert self.transform is not None, "None transform is not supported"
        self.skip_frames = skip_frames  # k, gap between o_t and o_t+k+1
        self.action_only = action_only
        # assert os.path.exists(data_root), "specified data_root does not exist"

        self.process_dataset(demo_dirs)

    # def get_img_dirs(self):
    #     image_extension = ".jpg"
    #     img_dirs = []
    #     for root, dirs, files in os.walk(self.data_root):
    #         for directory in dirs:
    #             dir_path = os.path.join(root, directory)
    #             if any(file.lower().endswith(image_extension) for file in os.listdir(dir_path)):
    #                 if not directory.startswith("depth"):  # ignores depth_images in bridge dataset
    #                     img_dirs.append(dir_path)

    #     return img_dirs

    def get_frame_no(self, filename):
        match = re.search(r"\d{1,}", filename)
        if match:
            frame_no = int(match.group())
        else:
            frame_no = -1
        return frame_no

    def get_img_paths(self, demo_dirs):
        frames_per_demo = []
        path_to_frames = []
        for folder_path in demo_dirs:
            frames = sorted(os.listdir(folder_path), key=self.get_frame_no)
            new_frames = []
            for frame in frames:
                if frame.endswith(".npy") or frame.endswith(".pkl"):
                    continue
                else:
                    new_frames.append(frame)
            path_to_frames.append(new_frames)
            num_frames = len(new_frames)
            frames_per_demo.append(num_frames)
        return path_to_frames, frames_per_demo

    def get_action_dim(self):
        # We need to know the shape of actions to create the correct tensors
        # Hence, we save the shapes in a dictionary for easy access and load it here
        action_dim = 1
        if os.path.exists(os.path.join(self.data_root, "shapes.yaml")):
            self.shapes_dict = yaml.load(
                open(os.path.join(self.data_root, "shapes.yaml"), "r"),
                Loader=yaml.FullLoader,
            )
            action_dim = self.shapes_dict["action_dim"]
        return action_dim

    def process_dataset(self, demo_dirs):

        print("Processing dataset...")

        # Go through each folder and count the number of frames
        self.path_to_folders = demo_dirs
        # self.path_to_frames = []
        # self.frames_per_demo = []
        # self.path_to_folders = self.get_img_dirs()
        self.path_to_frames, self.frames_per_demo = self.get_img_paths(self.path_to_folders)
        self.action_dim = self.get_action_dim()

        print(f"Dataset consists of {len(self.path_to_folders)} demo sequences")

    def _get_act_chunk(self, demo_idx, start_idx, chunk_size):
        actions = torch.zeros([chunk_size, self.action_dim], dtype=torch.float32)
        amask = torch.zeros_like(actions)
        action_path = os.path.join(self.path_to_folders[demo_idx], "actions.npy")
        if os.path.exists(action_path):
            actions_all = torch.from_numpy(np.load(action_path))
            # if fewer than chunk_size actions exist from start_ix, select whatever is left
            chunk_size = min(chunk_size, len(actions_all) - start_idx)
            actions[:chunk_size] = torch.from_numpy(np.load(action_path))[start_idx : start_idx + chunk_size]
            amask = torch.ones_like(actions)

        return actions, amask

    def _get_img(self, demo_idx, frame_idx):
        path = os.path.join(
            self.data_root,
            self.path_to_folders[demo_idx],
            self.path_to_frames[demo_idx][frame_idx],
        )
        return self.transform(Image.open(path))

    def _get_ee(self, demo_idx, frame_idx):
        ee_pos_path = os.path.join(self.path_to_folders[demo_idx], "ee_states.npy")
        if os.path.exists(ee_pos_path):
            ee_pos = torch.Tensor(np.load(ee_pos_path))[frame_idx]
        return ee_pos


class VisDemoDataset(VisDemoBase):

    def __init__(
            self, 
            data_root, 
            demo_dirs, 
            transform, 
            skip_frames=5, 
            action_only=False, 
            use_ee=False,
            action_chunk_len=5,
        ):

        super().__init__(data_root, demo_dirs, transform, skip_frames, action_only)
        self.action_chunk_len = action_chunk_len
        self.skip_frames = skip_frames
        self.use_ee = use_ee

        # Compute the length of the dataset
        # Number of o_t, o_t+k+1, o_g tuples in the dataset
        self.ntuples_per_demo = []
        length = 0
        self.index_to_demo_index = {}
        for i, frames in enumerate(self.frames_per_demo):
            # formula: demo_length = frames - seq_len + 1
            demo_length = frames - self.skip_frames - 1
            for j in range(demo_length):
                self.index_to_demo_index[length + j] = (i, j)
            length += demo_length
            self.ntuples_per_demo.append(demo_length)
        self.cumsum_ntuples_per_demo = np.cumsum(self.ntuples_per_demo)

    def __len__(self):
        return self.cumsum_ntuples_per_demo[-1]

    def __getitem__(self, index):
        i, j = self.index_to_demo_index[index]

        actions, amask = self._get_act_chunk(i, j, self.action_chunk_len)
        if self.action_only:
            return actions, amask
        elif self.use_ee:
            curr_img = self._get_img(i, j)
            goal_img = self._get_img(i, -1)
            ee_pos   = self._get_ee(i, j)
            return curr_img, goal_img, ee_pos, actions, amask
        else:
            curr_img = self._get_img(i, j)
            goal_img = self._get_img(i, -1)
            return curr_img, goal_img, actions, amask

    @property
    def action_shape(self):
        return (self.action_chunk_len, self.action_dim)


class SeqVisDemoDataset(VisDemoBase):

    def __init__(
        self,
        data_root,
        demo_dirs,
        transform=None,
        skip_frames=5,
        action_only=False,
        use_ee=False,
        seq_len=5,
        action_chunk_len=5,
    ):
        super().__init__(data_root, demo_dirs, transform, skip_frames, action_only)
        self.seq_len = seq_len
        self.action_chunk_len = action_chunk_len
        self.skip_frames = skip_frames
        self.use_ee = use_ee

    def __len__(self):
        return sum(self.frames_per_demo) // self.skip_frames
        # return len(self.path_to_folders)*50

    def __getitem__(self, index):
        index = None
        demo_idx = np.random.randint(0, len(self.path_to_folders))
        last_idx = np.random.randint(0, self.frames_per_demo[demo_idx])

        actions, amask = self._get_act_chunk(demo_idx, last_idx, self.action_chunk_len)

        if self.action_only:
            return actions, amask
        else:
            idx = last_idx
            img_seq = []
            ee_seq  = []
            while len(img_seq) < self.seq_len:
                if idx >= 0:
                    img = self._get_img(demo_idx, idx)
                    ee  = self._get_ee (demo_idx, idx)
                    idx -= self.skip_frames
                else:
                    if isinstance(img, list):
                        img = [torch.zeros_like(im) for im in img]
                    else:
                        img = torch.zeros_like(img)
                    ee  = torch.zeros_like(ee)
                img_seq.append(img)
                ee_seq .append(ee)
            # img_seq = torch.stack(img_seq)
            # data augmentation returns lists torch.stack(list(list)) fails
            goal_img = self._get_img(demo_idx, -1)

            if self.use_ee:
                return img_seq, goal_img, ee_seq, actions, amask
            else:
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

    B = 4
    T = 2
    A = 3

    dataset = SeqVisDemoDataset(data_root, transform, seq_len=T, action_chunk_len=A)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)

    for batch in dataloader:
        img_seq, goal, actions, amask = batch
        assert isinstance(img_seq, list)
        assert isinstance(img_seq[0], torch.Tensor)
        assert len(img_seq) == T
        assert img_seq[0].shape[0] == B
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
