import os
import pathlib
import pickle

import numpy as np
from torch.utils.data import DataLoader, Dataset


class Nav2DDataset(Dataset):
    """Nav2D Dataset"""

    def __init__(self, seq_len=4):
        self.seq_len = seq_len
        self.data_path = pathlib.Path(os.environ["DATA_ROOT"], "nav2d", "1000demos.pkl")
        data = pickle.load(open(self.data_path, "rb"))
        self.data = []
        # filter out demos that are too short
        for demo in data:
            start, goal, path, path_edges = demo
            if len(path) <= self.seq_len + 1:
                continue
            else:
                self.data.append((start, goal, path, path_edges))

        self.lengths = []
        self.indices = [0]
        self.seq_len = seq_len
        num_prev_paths = 0
        for demo in self.data:
            start, goal, path, path_edges = demo
            num_paths = len(path) - self.seq_len + 1
            self.lengths.append(num_prev_paths + num_paths)
            self.indices.append(self.indices[-1] + len(path))
            num_prev_paths += num_paths

        # print("indices", self.indices)

    def __getitem__(self, index):
        # find the demo index
        # print("index", index)

        demo_index = 0
        while True:
            if index < self.lengths[demo_index]:
                break
            else:
                demo_index += 1
        # debug info
        # print("demo_index", demo_index)
        # print("self.length[demo_index]", self.length[demo_index])
        # print("self.length[demo_index + 1]", self.length[demo_index + 1])
        # print("self.indices[demo_index]", self.indices[demo_index])
        # print("self.indices[demo_index + 1]", self.indices[demo_index + 1])

        # find demo index
        start, goal, path, path_edges = self.data[demo_index]

        # construct sequence of observation
        if demo_index == 0:
            idx = index
        else:
            idx = index - self.lengths[demo_index - 1]
        pos = path[idx : idx + self.seq_len, :]
        obs = np.concatenate([pos, np.broadcast_to(goal, (pos.shape[0], pos.shape[-1]))], axis=-1)

        actions = path_edges[idx : idx + self.seq_len - 1, :]

        return obs, actions

    def __len__(self):
        return self.lengths[-1]


class Nav2DDatasetV2(DataLoader):

    def __init__(self, skip_frames=0):
        # self.seq_len = seq_len
        self.skip_frames = skip_frames

        # Load data from pickle file
        self.data_path = pathlib.Path(os.environ["DATA_ROOT"], "nav2d", "1000demos.pkl")
        data = pickle.load(open(self.data_path, "rb"))
        self.data = []
        # filter out demos that are too short
        self.frames_per_demo = []
        for demo in data:
            start, goal, path, path_edges = demo
            if len(path) <= self.skip_frames + 1:
                continue
            else:
                self.data.append((start, goal, path, path_edges))
                self.frames_per_demo.append(len(path))

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

    def __len__(self) -> int:
        return self.cumsum_ntuples_per_demo[-1]

    def __getitem__(self, index):
        i, j = self.index_to_demo_index[index]
        path = self.data[i][2]
        obs = path[j, :]
        obs_next = path[j + self.skip_frames + 1, :]
        goal = path[-1, :]

        # import pudb

        # pudb.set_trace()
        actions = path[j + 1 : j + self.skip_frames + 2, :] - path[j : j + self.skip_frames + 1, :]
        return obs, obs_next, goal, actions


class Nav2DDataloader(DataLoader):
    """Nav2D Dataloader"""

    def __init__(self, skip_frames=2, batch_size=32, shuffle=True, num_workers=2):
        dataset = Nav2DDatasetV2(skip_frames=skip_frames)
        super(Nav2DDataloader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )


def test_dataloader():

    dataset = Nav2DDataset()

    print("The length of the dataset is ", len(dataset))

    dataloader = Nav2DDataloader()
    for i, batch in enumerate(dataloader):
        obs, actions = batch
        print("Obs shape :", obs.shape)
        print("Action shape :", actions.shape)
        break


def test_dataloader_v2():

    skip_frames = 4
    dataset = Nav2DDatasetV2(skip_frames=skip_frames)
    obs, obs_next, goal, actions = dataset[0]
    print("obs_next - obs", obs_next - obs)
    print("actions", actions)

    assert obs.shape[0] == 2, "obs shape is not correct"
    assert obs_next.shape[0] == 2, "obs_next shape is not correct"
    assert goal.shape[0] == 2, "goal shape is not correct"
    assert actions.shape[0] == skip_frames + 1, "actions shape is not correct"

    print("The length of the dataset is ", len(dataset))

    dataloader = Nav2DDataloader()
    for i, batch in enumerate(dataloader):
        obs, obs_next, goal, actions = batch
        print("Obs shape :", obs.shape)
        print("Action shape :", actions.shape)
        break


if __name__ == "__main__":
    test_dataloader_v2()
