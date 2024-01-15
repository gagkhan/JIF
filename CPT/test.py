import pylab
from PIL import Image
import imageio
from torch.utils.data import Dataset
from itertools import accumulate

import glob

def fetch_image():
    import imageio
    filename = 'data/ihm_ar/0.mp4'
    vid = imageio.get_reader(filename,  'ffmpeg')
    image = vid.get_data(10)
    image = Image.fromarray(image).resize((64, 64))
    fig = pylab.figure()
    fig.suptitle('image #{}'.format(0), fontsize=20)
    pylab.imshow(image)
    fig.savefig("tmp.png")
    pylab.show()
    
class IHMDataset(Dataset):
    
    def __init__(self) -> None:
        self.files = []
        self.nframes = []
        for name in glob.glob("data/ihm_ar/*.mp4"):
            self.files.append(name)
            vid = imageio.get_reader(name,  'ffmpeg')
            self.nframes.append(vid.count_frames())
        self.nframes_cum = list(accumulate(self.nframes))
                
    def __getitem__(self, index):
        for vid_idx, nframes_cum in enumerate(self.nframes_cum):
            if index < nframes_cum:
                break
        vid = imageio.get_reader(self.files[vid_idx])
        vid._meta["fps"] = 1000 # Encoded FPS property from QuickTime
        # print(index - self.nframes_cum[vid_idx-1])
        img = vid.get_data(index)
        # fig = pylab.figure()
        # fig.suptitle('image #{}'.format(0), fontsize=20)
        # pylab.imshow(img)
        # fig.savefig("tmp.png")
        
    def __len__(self):
        return self.nframes_cum[-1]
        

if __name__ == "__main__":
    
    ihmdataset = IHMDataset()
    
    print("Length of dataset:", len(ihmdataset))
    
    # create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(ihmdataset, batch_size=32, shuffle=True, num_workers=0)
    
    import time
    start = time.time()
    num_batches = 0
    for batch in dataloader:
        print(batch.shape)
        num_batches += 1
        print("Batch:", num_batches)
    end = time.time()
    tpb = (end - start) / num_batches
    
    print("Average time per batch:", tpb)
    
    
    
    
    
    
    
    
    