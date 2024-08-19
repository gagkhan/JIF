import argparse
import os
import random
import re

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_num(self, name):
    match = re.search(r"\d{1,}", name)
    num = -1
    if match:
        num = int(match.group())
    return num


def get_demo_dirs(data_root):
    """dirs with name demo_* are found"""
    prefix = "demo_"
    demo_dirs = []
    for subdir in os.listdir(data_root):
        if subdir.startswith(prefix):
            demo_dir = os.path.join(data_root, subdir)
            demo_dirs.append(demo_dir)
    return demo_dirs


class MultiModalDataset(Dataset):

    legal_keys = [
        "cam1",
        "cam2",
        "cam3",
        "tactile",
        "actions",
        "amask",
        "ee_pose",
    ]

    def __init__(
        self,
        root,
        demo_dirs,
        transform,
        keys=["cam1", "cam2", "tactile", "actions", "amask"],
        skip_frames=5,
    ):
        super().__init__()
        self.root = root
        self.demo_dirs = demo_dirs
        for key in keys:
            assert key in self.legal_keys, "unknown key specified"
        self.keys = keys
        self.skip_frames = skip_frames
        self.transform = transform
        self.chunk_size = self.skip_frames + 1

        self.frames_per_demo = []
        for demo in self.demo_dirs:
            # num_frames = len(os.listdir(os.path.join(demo, "cam1", "color")))
            num_frames = len(np.load(os.path.join(demo, "tactile.npy")))
            self.frames_per_demo.append(num_frames)
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
        self.action_dim = self.get_action_dim()
        self.action_shape = (self.skip_frames + 1, self.action_dim)

        self._fetch_val_fmap = {
            "cam1": self._get_img_cam1,
            "cam2": self._get_img_cam2,
            "cam3": self._get_img_cam3,
            "ee_pose": self._get_ee_pose,
            "tactile": self._get_tactile,
            "amask": self._get_act_mask,
            "actions": self._get_act_chunk,
        }

    def __len__(self):
        return self.cumsum_ntuples_per_demo[-1]

    def _get_img(self, cam, demo_idx, frame_idx):
        path = os.path.join(
            self.demo_dirs[demo_idx],
            cam,
            "color",
            "color_" + str(frame_idx).zfill(6) + ".png",
        )
        return self.transform(Image.open(path))

    def _get_img_cam1(self, demo_idx, frame_idx):
        imgs = {
            "cam1_curr": self._get_img("cam1", demo_idx, frame_idx),
            "cam1_next": self._get_img("cam1", demo_idx, frame_idx + self.skip_frames + 1),
            "cam1_goal": self._get_img("cam1", demo_idx, self.frames_per_demo[demo_idx] - 1),
        }
        return imgs

    def _get_img_cam2(self, demo_idx, frame_idx):
        imgs = {
            "cam2_curr": self._get_img("cam2", demo_idx, frame_idx),
            "cam2_next": self._get_img("cam2", demo_idx, frame_idx + self.skip_frames + 1),
            "cam2_goal": self._get_img("cam2", demo_idx, self.frames_per_demo[demo_idx] - 1),
        }
        return imgs

    def _get_img_cam3(self, demo_idx, frame_idx):
        imgs = {
            "cam3_curr": self._get_img("cam3", demo_idx, frame_idx),
            "cam3_next": self._get_img("cam3", demo_idx, frame_idx + self.skip_frames + 1),
            "cam3_goal": self._get_img("cam3", demo_idx, self.frames_per_demo[demo_idx] - 1),
        }
        return imgs

    def _get_ee_pose(self, demo_idx, frame_idx):
        """Get end effector state. Dimensionality differs between tasks.
        3-d for 3D goal reaching and so on.."""
        ee_path = os.path.join(self.demo_dirs[demo_idx], "ee_states.npy")
        if os.path.exists(ee_path):
            ee_pose = torch.Tensor(np.load(ee_path))[frame_idx]
        return {"ee_pose": ee_pose}

    def _get_tactile(self, demo_idx, frame_idx):
        path = os.path.join(self.demo_dirs[demo_idx], "tactile.npy")
        assert os.path.exists(path)
        goal_idex = max(0, self.frames_per_demo[demo_idx] - 2)
        tactile = {
            "tactile_curr": torch.Tensor(np.load(path))[frame_idx],
            "tactile_next": torch.Tensor(np.load(path))[frame_idx + self.skip_frames + 1],
            "tactile_goal": torch.Tensor(np.load(path))[goal_idex],
        }
        return tactile

    def _get_act_chunk(self, demo_idx, start_idx):
        chunk_size = self.chunk_size
        actions = torch.zeros([chunk_size, self.action_dim], dtype=torch.float32)
        action_path = os.path.join(self.demo_dirs[demo_idx], "actions.npy")
        if os.path.exists(action_path):
            actions_all = torch.from_numpy(np.load(action_path))
            # if fewer than chunk_size actions exist from start_ix, select whatever is left
            chunk_size = min(chunk_size, len(actions_all) - start_idx)
            actions[:chunk_size] = torch.from_numpy(np.load(action_path))[start_idx : start_idx + chunk_size]
        return {"actions": actions}

    def _get_act_mask(self, demo_idx, start_idx):
        chunk_size = self.chunk_size
        amask = torch.zeros([chunk_size, self.action_dim], dtype=torch.float32)
        action_path = os.path.join(self.demo_dirs[demo_idx], "actions.npy")
        if os.path.exists(action_path):
            amask = torch.ones_like(amask)
        return {"amask": amask}

    def get_action_dim(self):
        # We need to know the shape of actions to create the correct tensors
        # Hence, we save the shapes in a dictionary for easy access and load it here
        action_dim = 1
        if os.path.exists(os.path.join(self.root, "shapes.yaml")):
            self.shapes_dict = yaml.load(
                open(os.path.join(self.root, "shapes.yaml"), "r"),
                Loader=yaml.FullLoader,
            )
            action_dim = self.shapes_dict["action_dim"]
        return action_dim

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

    def __getitem__(self, index):
        demo_idx, frame_idx = self.index_to_demo_index[index]
        item = dict()
        for key in self.keys:
            item.update(self._fetch_val_fmap[key](demo_idx, frame_idx))
        return item
    
def datakeys(args):
    keys = ["cam1", "tactile", "actions", "amask"]
    if args.use_cam2:
        keys.append("cam2")
    if args.use_cam3:
        keys.append("cam3")
    if hasattr(args, "use_ee"):
        if args.use_ee:
            keys.append("ee_pose")
    if hasattr(args, "action_only"):
        if args.__dict__["action_only"]:
            keys = ["actions"]
    return keys


def load_dataset(args, keys, transform=None):

    data_root = args.data_path
    train_split = args.train_split

    assert os.path.exists(data_root), "specified data_root does not exist"
    assert transform is not None, "None transform is not supported"
    demo_dirs = get_demo_dirs(data_root)
    random.shuffle(demo_dirs)

    print(f"Number of demos: {len(demo_dirs)}")

    train_dirs = demo_dirs[: int(train_split * len(demo_dirs))]
    val_dirs = demo_dirs[int(train_split * len(demo_dirs)) :]

    kwargs = dict()
    for param in ["skip_frames"]:
        if hasattr(args, param):
            kwargs[param] = args.__dict__[param]

    train_dataset = MultiModalDataset(data_root, train_dirs, transform, keys=keys, **kwargs)
    val_dataset = MultiModalDataset(data_root, val_dirs, transform, keys=keys, **kwargs)

    return train_dataset, val_dataset


if __name__ == "__main__":

    data_root = "/ssd01/gagan/cpt_data/ours/aug09_pickhuman"
    # demo_dirs = get_demo_dirs(data_root)
    # for d in demo_dirs:
    #     print(d)

    # dataset = MultiModalDataset("/ssd01/gagan/cpt_data/ours/aug09_pickhuman")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default=data_root,
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument(
        "--skip_frames",
        default=5,
        type=int,
        help="Number of frames to skip when loading the dataset.",
    )
    parser.add_argument(
        "--train_split",
        default=0.9,
        type=float,
        help="split fraction of data for training, rest is used for validation",
    )

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    train_dataset, val_dataset = load_dataset(parser.parse_args(), transform=transform)

    dataitem = train_dataset[0]

    print(dataitem.keys())

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        # sampler=torch.utils.data.DistributedSampler(train_dataset, shuffle=True),
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for batch in data_loader:
        print(type(batch))
        print(batch["cam1_curr"].shape)
        print(batch["tact_curr"].shape)
        break
