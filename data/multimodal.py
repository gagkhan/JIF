import os
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_num(self, name):
    match = re.search(r"\d{1,}", name)
    num = -1
    if match: num = int(match.group()) 
    return num

def get_demo_dirs(data_root):
    """dirs with name demo_* are found """
    prefix = "demo_"
    demo_dirs = []
    for subdir in os.listdir(data_root):
        if subdir.startswith(prefix):
            demo_dir = os.path.join(data_root, subdir)
            demo_dirs.append(demo_dir)
    return demo_dirs


class MultiModalDataset(Dataset):
    
    legal_keys = ["cam0", "cam1", "cam2", "tactile", "actions", "amask", "ee_state"]
    
    def __init__(self, root, demo_dirs, keys=None):
        super().__init__()
        self.root = root
        self.demo_dirs = demo_dirs
        self.keys = ["cam0", "cam1", "tactile"]
        self._fetch_val_fmap = {
            "cam0" : self._get_img_cam0,
            "cam1" : self._get_img_cam1,
            "cam2" : self._get_img_cam2,
            "ee_state": self._get_ee,
            "tactile": self._get_tactile,
            "amask": self._get_act_mask,
            "actions": self._get_actions,
        }
   
    def _get_img(self, cam, demo_idx, frame_idx):
        path = os.path.join(
            self.root,
            self.path_to_demos[demo_idx],
            cam,
            self.path_to_frames[demo_idx][frame_idx],
        )
        return self.transform(Image.open(path))
    
    def _get_img_cam0(self, demo_idx, frame_idx)
        return _get_img("cam0", demo_idx, frame_idx)

    def _get_img_cam0(self, demo_idx, frame_idx)
        return _get_img("cam1", demo_idx, frame_idx)
    
    def _get_img_cam0(self, demo_idx, frame_idx)
        return _get_img("cam2", demo_idx, frame_idx)
    
    def _get_ee(self, demo_idx, frame_idx):
        ee_pos_path = os.path.join(self.path_to_folders[demo_idx], "ee_states.npy")
        if os.path.exists(ee_pos_path):
            ee_pos = torch.Tensor(np.load(ee_pos_path))[frame_idx]
        return ee_pos
    
    def _get_tactile(self, demo_idx, frame_idx):
        ee_pos_path = os.path.join(self.path_to_folders[demo_idx], "tactile.npy")
        if os.path.exists(ee_pos_path):
            ee_pos = torch.Tensor(np.load(ee_pos_path))[frame_idx]
        return ee_pos
    
    def _get_act_chunk(self, demo_idx, start_idx, chunk_size):
        actions = torch.zeros([chunk_size, self.action_dim], dtype=torch.float32)
        action_path = os.path.join(self.path_to_folders[demo_idx], "actions.npy")
        if os.path.exists(action_path):
            actions_all = torch.from_numpy(np.load(action_path))
            # if fewer than chunk_size actions exist from start_ix, select whatever is left
            chunk_size = min(chunk_size, len(actions_all) - start_idx)
            actions[:chunk_size] = torch.from_numpy(np.load(action_path))[start_idx : start_idx + chunk_size]
        return actions
    
    def _get_act_mask(self, demo_idx, start_idx, chunk_size):
        amask = torch.zeros_like(actions)
        action_path = os.path.join(self.path_to_folders[demo_idx], "actions.npy")
        if os.path.exists(action_path):
            amask = torch.ones_like(actions)
        return amask
    
    def _fetch_val(key, demo_idx, frame_idx):
        if "cam" in key:
            return self._get_img(key, demo_idx, frame_idx)
        elif key == "ee_states":
            return self._get_ee(demo_idx, frame_idx)
        elif key == "tactile":
            return self._get_tactile(demo_idx, frame_idx)
        elif key == "actions":
            return self._get_act_chunk(demo_idx, frame_idx)
        elif key == "amask":
            return self._get_act_chunk(demo_idx, frame_idx)
        
    def __getitem__(self, index):
        demo_idx, frame_idx = 0, 0
        item = { key: self._fetch_val_fmap[key](demo_idx, frame_idx)}
        return item 
        
        
    


if __name__ == "__main__":
    
    data_root = "/ssd01/gagan/cpt_data/ours/aug09_pickhuman"
    demo_dirs = get_demo_dirs(data_root)
    for d in demo_dirs:
        print(d)
        
    
    dataset = MultiModalDataset("/ssd01/gagan/cpt_data/ours/aug09_pickhuman")
    
    
    
    
    