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


def test_dataloader():

    dataset = Nav2DDataset()

    print("The length of the dataset is ", len(dataset))

    dataloader = Nav2DDataloader()
    for i, batch in enumerate(dataloader):
        obs, actions = batch
        print("Obs shape :", obs.shape)
        print("Action shape :", actions.shape)
        break


if __name__ == "__main__":
    test_dataloader()
