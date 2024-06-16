import os
import pathlib
import pickle

import numpy as np
from torch.utils import data


class Dataset(data.Dataset):

    def __init__(self, root, skip_frames=0):
        """
        Args:
            root: str
            skip_frames: int (default: 0)
        """
        self.skip_frames = skip_frames
        data = self._load_data(root)
        self._process_data(data)

    def _load_data(self, root: str):
        """
        Args:
            root: str
        Returns:
            data: list [[start: np.ndarray, goal: np.ndarray, path: np.ndarray, actions: np.ndarray], ...]
        """
        data = pickle.load(open(root, "rb"))

        return data

    def _process_data(self, data):
        """
        Summary: Process the data and filter out demos that are too short
        Args:
            data: list [[start: np.ndarray, goal: np.ndarray, path: np.ndarray, actions: np.ndarray], ...]
        Returns:
            None
        """

        # filter out demos that are too short
        self.frames_per_demo = []
        self.data = []
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
        actions = path[j + 1 : j + self.skip_frames + 2, :] - path[j : j + self.skip_frames + 1, :]
        return obs, obs_next, goal, actions


class Dataloader(data.DataLoader):
    """Dataloader"""

    def __init__(self, root, skip_frames=2, batch_size=32, shuffle=True, num_workers=2):
        dataset = Dataset(root=root, skip_frames=skip_frames)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run():

    skip_frames = 4
    data_root = pathlib.Path(os.environ["DATA_ROOT"], "nav2d", "demos.pkl")
    dataset = Dataset(data_root, skip_frames=skip_frames)
    obs, obs_next, goal, actions = dataset[0]
    print("obs_next - obs", obs_next - obs)
    print("actions", actions)

    assert obs.shape[0] == 2, "obs shape is not correct"
    assert obs_next.shape[0] == 2, "obs_next shape is not correct"
    assert goal.shape[0] == 2, "goal shape is not correct"
    assert actions.shape[0] == skip_frames + 1, "actions shape is not correct"

    print("The length of the dataset is ", len(dataset))

    dataloader = Dataloader(root=data_root)
    for i, batch in enumerate(dataloader):
        obs, obs_next, goal, actions = batch
        print("Obs shape :", obs.shape)
        print("Action shape :", actions.shape)
        break


if __name__ == "__main__":
    test()
