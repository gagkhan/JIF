#@markdown ### **Imports**
# diffusion policy import
import argparse
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import wandb

# env import
from PIL import Image
import os



########################################################################################
#@markdown ### **Arg Parser**

def get_args_parser():
    parser = argparse.ArgumentParser("CPT-diffusion", add_help=False)

    # Dataset
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/train/",
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument(
        "--pred_horizon",
        default=32,
        type=int,
        help="""Prediction horizon""",
    )
    parser.add_argument(
        "--obs_horizon",
        default=16,
        type=int,
        help="""Observation horizon""",
    )
    parser.add_argument(
        "--action_horizon",
        default=17,
        type=int,
        help="""Action horizon (<= 1 + pred_horizon - obs_horizon)""",
    )

    # Training
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="""Number of epochs of training""",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="""Batch size""",
    )
    parser.add_argument(
        "--use_tactile",
        action='store_true',
        help="""Use tactile data""",
    )

    return parser



########################################################################################
#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTImageDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data ((image, agent_pos), action) from a zarr storage
#@markdown - Normalizes each dimension of agent_pos and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
#@markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
#@markdown  - key `action`: shape (pred_horizon, 2)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        obs_horizon: int):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = start_idx - (obs_horizon - 1)
        max_start = end_idx   - (obs_horizon + 1)

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            indices.append({
                'o': [max(start_idx, min(e, end_idx-1)) for e in range(idx, idx+obs_horizon+1)],
                'p': [max(start_idx, min(e, end_idx-1)) for e in range(idx, idx+sequence_length)],
                'g': end_idx-1
            })
    indices = np.array(indices)
    return indices

def get_demo_dirs(data_root):
    """dirs with name demo_* are found"""
    prefix = "demo_"
    demo_dirs = []
    for subdir in os.listdir(data_root):
        if subdir.startswith(prefix):
            demo_dir = os.path.join(data_root, subdir)
            demo_dirs.append(demo_dir)
    return demo_dirs

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        self.demo_dirs = get_demo_dirs(dataset_path)

        frames_per_demo = []
        self.index_to_demo_index = []
        train_data = {
            "tactile":   [], # (N, 2)
            "agent_pos": [], # (N, 8)
            "action":    [], # (N, 8)
        }
        for demo_idx, demo in enumerate(self.demo_dirs):
            # frames_per_demo
            num_frames = len(os.listdir(os.path.join(demo, "cam1", "color")))
            frames_per_demo.append(num_frames)

            # index_to_demo_index
            idx_list = list(zip([demo_idx]*num_frames, range(num_frames)))
            self.index_to_demo_index.extend(idx_list)

            # tactile
            tactile = np.load(os.path.join(demo, "tactile.npy"))
            train_data['tactile'].append(tactile)

            # agent_pos
            agent_pos = np.load(os.path.join(demo, "ee_states.npy"))
            train_data['agent_pos'].append(agent_pos)

            # action
            action = np.load(os.path.join(demo, "commands.npy"))
            train_data['action'].append(action)

        train_data['tactile']   = np.concatenate(train_data['tactile'])
        train_data['agent_pos'] = np.concatenate(train_data['agent_pos'])
        train_data['action']    = np.concatenate(train_data['action'])

        episode_ends = np.cumsum(frames_per_demo)

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            obs_horizon=obs_horizon)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images will be loaded during training
        normalized_train_data.update({
            "cam1":    None, # (N, (3, 224, 224)) 
            "cam2":    None, # (N, (3, 224, 224))
            "cam3":    None, # (N, (3, 224, 224))
        })

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def _get_img(self, cam, demo_idx, frame_idx):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.9,1.0), ratio=(1.3,1.4), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        path = os.path.join(
            self.demo_dirs[demo_idx],
            cam,
            "color",
            "color_" + str(frame_idx).zfill(6) + ".png",
        )
        return transform(Image.open(path))

    def __getitem__(self, idx):

        # get the start/end indices for this datapoint
        indices = self.indices[idx]
        train_data = self.normalized_train_data

        # get data
        nsample = dict()
        nsample['cam1'] = torch.stack([self._get_img('cam1', *(self.index_to_demo_index[i])) for i in indices['o']])
        nsample['cam2'] = torch.stack([self._get_img('cam2', *(self.index_to_demo_index[i])) for i in indices['o']])
        nsample['cam3'] = torch.stack([self._get_img('cam3', *(self.index_to_demo_index[i])) for i in indices['o']])
        nsample['tactile'  ] = torch.tensor(train_data['tactile'  ][indices['o']], dtype=torch.float32)
        nsample['agent_pos'] = torch.tensor(train_data['agent_pos'][indices['o']], dtype=torch.float32)
        nsample['action'   ] = torch.tensor(train_data['action'   ][indices['p']], dtype=torch.float32)

        # nsample['cam1_goal'] = self._get_img('cam1', *(self.index_to_demo_index[indices['g']]))
        # nsample['cam2_goal'] = self._get_img('cam2', *(self.index_to_demo_index[indices['g']]))
        # nsample['cam3_goal'] = self._get_img('cam3', *(self.index_to_demo_index[indices['g']]))
        # nsample['tactile_goal'] = torch.tensor(train_data['tactile'][indices['g']], dtype=torch.float32)

        return nsample



########################################################################################
#@markdown ### **Dataset Demo**

def dataset_demo(args):
    # args
    dataset_path = args.data_path
    batch_size = args.batch_size

    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon
    assert(pred_horizon == obs_horizon+action_horizon-1)
    #|o|o|o|o|o|o|o|o|o|n              observations
    #|               |a|a|a|a|a|a|a|a| actions executed
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted

    # create dataset from file
    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=32,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['cam1'].shape:      ", batch['cam1'].shape)      # (B, obs_horiz+1, 3, 224, 224)
    print("batch['cam1'].dtype:      ", batch['cam1'].dtype)
    print("batch['tactile'].shape:   ", batch['tactile'].shape)   # (B, obs_horiz+1, 2)
    print("batch['tactile'].dtype:   ", batch['tactile'].dtype)
    print("batch['agent_pos'].shape: ", batch['agent_pos'].shape) # (B, obs_horiz+1, 8)
    print("batch['agent_pos'].dtype: ", batch['agent_pos'].dtype)
    print("batch['action'].shape:    ", batch['action'].shape)    # (B, pred_horiz, 8)
    print("batch['action'].dtype:    ", batch['action'].dtype)

    return dataloader



########################################################################################
#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


########################################################################################
#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def get_vitact(use_tactile) -> nn.Module:
    """
    name: no use, discarded
    weights: "IMAGENET1K_V1", None
    """
    from visuotactile.utils import build_vitact_encoder

    encoder_args = argparse.Namespace(
        encoder_arch='vitact_tiny',
        use_tactile =use_tactile,
        use_cam2    =True,
        use_cam3    =True,
        patch_size  =None,
        drop_path_rate=0.1)
    # checkpoint = torch.load("checkpoint_best.pth", map_location="cpu")
    # encoder_args = checkpoint["args"]
    # state_dict = {k.replace("module.encoder.", ""): v for k, v in checkpoint["student"].items() if "module.encoder." in k}

    encoder, vision_feature_dim = build_vitact_encoder(encoder_args)
    # encoder.load_state_dict(state_dict)
    # for p in encoder.parameters():
    #     p.requires_grad = False
    # encoder.eval()

    return encoder, encoder_args, vision_feature_dim


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module



########################################################################################
#@markdown ### **Network Demo**

def network_demo(args):
    # args
    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon

    use_tactile = args.use_tactile

    # construct encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder1, encoder_args, vision_feature_dim = get_vitact(use_tactile)

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder1 = replace_bn_with_gn(vision_encoder1)

    # Encoder has output dim of this
    vision_feature_dim = vision_feature_dim
    # agent_pos is 8 dimensional
    lowdim_obs_dim = 8
    # observation feature has these dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 8

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder1': vision_encoder1,
        'noise_pred_net': noise_pred_net
    })

    # demo
    with torch.no_grad():
        # example inputs
        image = torch.zeros((1, obs_horizon,3,224,224))
        tacile = torch.zeros((1, obs_horizon, 2))
        agent_pos = torch.zeros((1, obs_horizon, 8))
        # vision encoder
        image_features1 = nets['vision_encoder1']([
            image.flatten(end_dim=1), \
            *([tacile.flatten(end_dim=1)] if use_tactile else []),
            image.flatten(end_dim=1), \
            image.flatten(end_dim=1)])

        image_features1 = image_features1.reshape(*image.shape[:2],-1)
        # (1,obs_horiz,D)
        obs = torch.cat([image_features1, agent_pos],dim=-1)
        # (1,obs_horiz,D+8)

        noised_action = torch.randn((1, pred_horizon, action_dim))
        diffusion_iter = torch.zeros((1,))
        # (1,pred_horiz,action_dim)

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = nets['noise_pred_net'](
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))

        # illustration of removing noise
        # the actual noise removal is performed by NoiseScheduler
        # and is dependent on the diffusion noise schedule
        denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # # load pretrained noise_pred_net
    # checkpoint = torch.load("/ssd01/gagan/cpt_checkpoints/12_01_diff_v0.3/checkpoint_best.pth")
    # state_dict = {k.replace("noise_pred_net.", ""): v for k, v in checkpoint["ema_nets"].items() if "noise_pred_net." in k}
    # nets['noise_pred_net'].load_state_dict(state_dict)

    # device transfer
    device = torch.device('cuda')
    _ = nets.to(device)

    # visualize data in batch
    print("image_features1.shape:  ", image_features1.shape) # (B,obs_horiz,D)
    print("obs.shape:              ", obs.shape)             # (B,obs_horiz,D+8)
    print("noised_action.shape     ", noised_action.shape)   # (B,pred_horiz,action_dim)
    print("noise.shape:            ", noise.shape)           # (B,pred_horiz,action_dim)

    return nets, encoder_args, num_diffusion_iters, noise_scheduler, device



########################################################################################
#@markdown ### **Training**
#@markdown
#@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

def training(args, dataloader, nets, encoder_args, num_diffusion_iters, noise_scheduler, device):
    # args
    num_epochs = args.num_epochs
    use_tactile = encoder_args.use_tactile

    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        best_loss = 999
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage1 = nbatch['cam1'][:,:obs_horizon].to(device)
                    nimage2 = nbatch['cam2'][:,:obs_horizon].to(device)
                    nimage3 = nbatch['cam3'][:,:obs_horizon].to(device)

                    ntactile = nbatch['tactile'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features1 = nets['vision_encoder1']([
                        nimage1.flatten(end_dim=1), \
                        *([ntactile.flatten(end_dim=1)] if use_tactile else []),
                        nimage2.flatten(end_dim=1), \
                        nimage3.flatten(end_dim=1)])
                    image_features1 = image_features1.reshape(
                        *nimage1.shape[:2],-1)
                    # (B,obs_horizon,D)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat( \
                        [image_features1, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B,obs_horizon*obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = nets["noise_pred_net"](
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            wandb.log({"loss": np.mean(epoch_loss)})


            ########################################################################################
            # Save model

            # Weights of the EMA model
            # is used for inference
            ema_nets = nets
            ema.copy_to(ema_nets.parameters())

            chkpnt = {
                "ema_nets" : ema_nets.state_dict(),
                "stats" : dataloader.dataset.stats,
                "obs_horizon" : obs_horizon,
                "action_horizon" : action_horizon,
                "pred_horizon" : pred_horizon,
                "num_diffusion_iters" : num_diffusion_iters,
                "encoder_args": encoder_args
            }

            if np.mean(epoch_loss) < best_loss:
                best_loss = np.mean(epoch_loss)
                torch.save(chkpnt, '/ssd01/gagan/cpt_checkpoints/diff/checkpoint_best.pth')

            if (epoch_idx+1) % 10 == 0 or (epoch_idx+1) == num_epochs:
                torch.save(chkpnt, f'/ssd01/gagan/cpt_checkpoints/diff/checkpoint_e{epoch_idx}.pth')

            torch.save(chkpnt, '/ssd01/gagan/cpt_checkpoints/diff/checkpoint_latest.pth')



########################################################################################
#@markdown ### **Main**

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    run=wandb.init(project="CPT", name="diff", config=args)

    dataloader = \
        dataset_demo(args)
    nets, encoder_args, num_diffusion_iters, noise_scheduler, device = \
        network_demo(args)

    training(args, dataloader, nets, encoder_args, num_diffusion_iters, noise_scheduler, device)

    run.finish()
    del dataloader
    torch.cuda.empty_cache()