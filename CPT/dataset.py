import pylab
from PIL import Image
import imageio
from torch.utils.data import Dataset
from itertools import accumulate

import glob

import torchvision.transforms as transforms
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1, transform=None):
        self.video_paths = video_paths
        self.video_clips = VideoClips(video_paths, clip_length_in_frames, frames_between_clips, frame_rate=16, num_workers=16)
        self.transform = transform

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        # Perform any additional processing or transformations here
        # if self.transform:
        #     video = self.transform(video)

        return video, video_idx


def test_video_dataset():
    video_paths = glob.glob("data/ihm_ar/*.mp4")
    video_dataset = VideoDataset(video_paths, clip_length_in_frames=16, frames_between_clips=1, transform=transforms.ToTensor())
    print("Length of dataset:", len(video_dataset))
    
    # create dataloader
    dataloader = DataLoader(video_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    import time
    start = time.time()
    for i, (video, video_idx) in enumerate(dataloader):
        print(i, video.shape, video_idx)
        if i == 10:
            break
    print("Time taken:", time.time() - start)
    

if __name__ == "__main__":
    
    test_video_dataset()