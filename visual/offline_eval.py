# This script for offline evaluation of the model
# We will load the demonstration trajectories and the model and evaluate the model
# on these trajectories.


# how to load the demonstration data as trajectories
# what is the input?
# what is the output?

# input is the path to the robot demonstration data

# iterator over the trajetories (sequence of images, sequence of actions)

# for each trajectory, we will:
# 1. load the images and the actions
# 2. run the model on the images
# 3. compare the predicted actions with the actual actions
# 4. report the error

# What to implement?
# A dataset class that loads the demonstration data

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import visual.vision_transformer as vits
from matplotlib.patches import Polygon
from PIL import Image
from torchvision import transforms
from visual.ilpo import MLP, ActionDecoder, Dynamics, ILPOWrapper, Policy


class SequenceDataset:
    """
    A dataset class for loading sequence data.

    Args:
        data_root (str): The root directory of the data.

    Attributes:
        data_root (str): The root directory of the data.
        data (list): A list of directories containing the data.
        transform (ToTensor): An instance of the ToTensor class for transforming the data.

    Methods:
        load_data(): Loads the data from the data root directory.
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.data = self.load_data()

        print(f"Loaded {len(self.data)} demonstrations")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def load_data(self):
        """
        Loads the data from the data root directory.

        Returns:
            list: A list of directories containing the data.

        """
        data = []
        for demo in os.listdir(self.data_root):
            if os.path.isdir(os.path.join(self.data_root, demo)):
                data.append(demo)
        return data

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the observations and actions.

        """
        # open the demo folder
        demo = self.data[idx]
        demo_path = os.path.join(self.data_root, demo)
        obs = []
        image_paths = []
        for frame in sorted(os.listdir(demo_path)):
            if frame.endswith(".jpg") or frame.endswith(".png"):
                image_paths.append(frame)
        for image_path in image_paths:
            image = self.transform(Image.open(os.path.join(demo_path, image_path)))
            obs.append(image)
        actions = np.load(os.path.join(demo_path, "actions.npy"))

        obs = torch.stack(obs)
        actions = torch.tensor(actions, dtype=torch.float32)

        return obs, actions


def load_model(args, device):
    """Load model from checkpoint."""

    assert os.path.isfile(args.pretrained_weights), "No pretrained weights found or file invalid"

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")

    model_args = state_dict["args"]

    # build vit
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    embed_dim = model.embed_dim
    if args.algo == "cpt":
        head = vits.DINOHead(embed_dim, model_args.out_dim, model_args.use_bn_in_head)
        # build ILPO student
        model = ILPOWrapper(
            model,
            head,
            embed_dim=embed_dim,
            latent_action_dim=model_args.latent_action_dim,
            policy_units=model_args.policy_units,
            dynamics_units=model_args.dynamics_units,
        )

        action_decoder = ActionDecoder(
            latent_action_dim=model_args.latent_action_dim,
            units=model_args.action_decoder_units,
            action_shape=(model_args.skip_frames + 1, 3),
        )

    elif args.algo == "bc":
        action_decoder = ActionDecoder(
            2 * embed_dim,
            units=model_args.action_decoder_units,
            action_shape=(model_args.skip_frames + 1, 3),
        )

    # load weights
    model_dict = state_dict[args.checkpoint_key]
    model_dict = {k.replace("module.", ""): v for k, v in model_dict.items()}
    model_dict = {k.replace("backbone.", ""): v for k, v in model_dict.items()}

    model.load_state_dict(model_dict)
    action_decoder.load_state_dict(state_dict["action_decoder"])

    for p in model.parameters():
        p.requires_grad = False
    for p in action_decoder.parameters():
        p.requires_grad = False
    model.eval()
    action_decoder.eval()
    model.to(device)
    action_decoder.to(device)

    return model, action_decoder


def get_arg_parser():
    """Get argument parser."""

    parser = argparse.ArgumentParser("Evaluate model for Visual Navigation.")

    parser.add_argument(
        "--data_root",
        default="/home/gagan/Home/VideoIL/data/ours/MoveT",
        type=str,
        help="The root directory of the data.",
    )

    parser.add_argument(
        "--arch",
        default="vit_tiny",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )

    parser.add_argument(
        "--algo",
        default="cpt",
        type=str,
        choices=["bc", "cpt"],
        help="Training algorithm with which the checkpoint was generated.",
    )

    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    parser.add_argument("--pretrained_weights", default="", type=str, help="Path to pretrained weights to load.")
    parser.add_argument(
        "--checkpoint_key",
        default="student",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--output_dir", default=".", help="Path where to save visualizations.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""",
    )
    return parser


def run_offline_evaluation_cpt(
    data_root,
    model,
    action_decoder,
    device,
):
    """
    Run offline evaluation on a given dataset using the ILPO model.

    Args:
        data_root (str): The root directory of the dataset.
        model: The ILPO model.
        action_decoder: The action decoder.
        device: The device to run the evaluation on.

    Returns:
        None
    """

    dataset = SequenceDataset(data_root)

    errors = []
    for ep in range(len(dataset)):
        obs, actions = dataset[ep]
        obs = obs.to(device)
        actions = actions.to(device)

        # get latent actions from the ILPO model
        error = 0
        for t in range(len(obs)):
            _, zt, zmu, zsigma = model(obs[t].unsqueeze(0), obs[-1].unsqueeze(0))
            # get the action from the action decoder
            action = action_decoder(zt)
            et = (action[0] - actions[t]).cpu().numpy()
            error += np.mean(et * et)
        error /= len(obs)

        print(f"Episode: {ep}, Error: {error}")
        errors.append(error)

    errors = np.array(errors)
    return round(np.mean(errors), 5), round(np.std(errors), 5)


def run_offline_evaluation_bc(
    data_root,
    model,
    action_decoder,
    device,
):
    """
    Run offline evaluation on a given dataset using the ILPO model.

    Args:
        data_root (str): The root directory of the dataset.
        model: The ILPO model.
        action_decoder: The action decoder.
        device: The device to run the evaluation on.

    Returns:
        None
    """

    dataset = SequenceDataset(data_root)

    errors = []
    for ep in range(len(dataset)):
        obs, actions = dataset[ep]
        obs = obs.to(device)
        actions = actions.to(device)

        # get latent actions from the ILPO model
        error = 0
        emb_g = model(obs[-1].unsqueeze(0))
        for t in range(len(obs)):
            emb_t = model(obs[t].unsqueeze(0))
            # get the action from the action decoder
            action = action_decoder(torch.cat([emb_t, emb_g], dim=-1))
            et = (action[0] - actions[t]).cpu().numpy()
            error += np.mean(et * et)
        error /= len(obs)
        print(f"Episode: {ep}, Error: {error}")
        errors.append(error)

    errors = np.array(errors)
    return round(np.mean(errors), 5), round(np.std(errors), 5)


if __name__ == "__main__":

    argparser = get_arg_parser()
    args = argparser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, action_decoder = load_model(args, device)

    if args.algo == "bc":
        error_mean, error_std = run_offline_evaluation_bc(args.data_root, model, action_decoder, device)
    elif args.algo == "cpt":
        error_mean, error_std = run_offline_evaluation_cpt(args.data_root, model, action_decoder, device)

    print(f"Error Mean: {error_mean}")
    print(f"Error Variance: {error_std}")
