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
