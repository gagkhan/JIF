import argparse
import colorsys
import os
import random
import sys
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np

# import requests
# import skimage.io
import torch
import torch.nn as nn

# import torchvision
# import visual.utils
import visual.vision_transformer as vits
from matplotlib.patches import Polygon
from PIL import Image
from skimage.measure import find_contours
from tasks.nav2d.nav2d import Map2D, Robot
from torchvision import transforms as pth_transforms
from visual.ilpo import MLP, ActionDecoder, Dynamics, ILPOWrapper, Policy


def get_arg_parser():
    """Get argument parser."""

    parser = argparse.ArgumentParser("Evaluate model for Visual Navigation.")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument("--patch_size", default=8, type=int, help="Patch resolution of the model.")
    parser.add_argument(
        "--pretrained_weights", default="", type=str, help="Path to pretrained weights to load."
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument(
        "--image_size", default=(480, 480), type=int, nargs="+", help="Resize image."
    )
    parser.add_argument("--output_dir", default=".", help="Path where to save visualizations.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""",
    )
    parser.add_argument(
        "--smooth_action",
        action="store_true",
        help="Exponential smoothing of the actions to reduce jitter. By default, it is False. It uses 0.9 smoothing factor.",
    )

    return parser


def load_model(args, device):
    """Load model from checkpoint."""

    assert os.path.isfile(args.pretrained_weights), "No pretrained weights found or file invalid"

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")

    model_args = state_dict["args"]

    # build vit
    transformer = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    embed_dim = transformer.embed_dim
    head = vits.DINOHead(embed_dim, model_args.out_dim, model_args.use_bn_in_head)

    # build ILPO student
    model = ILPOWrapper(
        transformer,
        head,
        embed_dim=embed_dim,
        latent_action_dim=model_args.latent_action_dim,
        units=model_args.policy,
    )

    action_decoder = ActionDecoder(
        model_args.latent_action_dim,
        units=model_args.action_decoder,
        action_shape=(model_args.skip_frames + 1, 2),
    )

    # load weights
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict["student"], strict=False)

    action_decoder.load_state_dict(state_dict["action_decoder"], strict=False)

    for p in model.parameters():
        p.requires_grad = False
    for p in action_decoder.parameters():
        p.requires_grad = False
    model.eval()
    action_decoder.eval()
    model.to(device)
    action_decoder.to(device)

    # curr_file = "/home/gagan/Home/VideoIL/data/nav2d_visual/0/000000.jpg"
    # goal_file = "/home/gagan/Home/VideoIL/data/nav2d_visual/0/000004.jpg"

    # def get_image_tensor(file):
    #     image = Image.open(file)
    #     image = image.resize(args.image_size)
    #     image = pth_transforms.ToTensor()(image).unsqueeze(0)
    #     image = image.to(device)
    #     return image

    # ot = get_image_tensor(curr_file)
    # og = get_image_tensor(goal_file)

    # out, zt, zmu, zsigma = model(ot, og)

    # action = action_decoder(zt)

    # zt = zt.detach().cpu().numpy()

    # print("latent action", zt)
    # print("action", action)

    return model, action_decoder, model_args


def get_image_tensor(file, image_size, device):
    image = Image.open(file)
    image = image.resize(image_size)
    image = pth_transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    return image


def evaluation(model, action_decoder, args, model_args):

    # Create map and robot
    print("Creating map and robot...")
    map = Map2D()
    # map.obstacle_radius = 0.125  # shrink obstacles to simulate padding
    robot = Robot(map)

    print("Testing model...")
    robot.reset()

    action_buffer = np.zeros((model_args.skip_frames + 1, 2))

    # before evaluating the model, lets create the temporary directory where we will save the images
    os.makedirs("/tmp/cpt", exist_ok=True)

    while True:
        # reset the robot to a random start and goal position
        robot.reset()
        start = robot.start
        goal = robot.goal

        # first move the robot to goal position to get the goal image
        robot.pos = goal
        robot.render()
        robot.fig.savefig(f"/tmp/cpt/visual_nav2d_goal.jpg")
        goal_image = get_image_tensor("/tmp/cpt/visual_nav2d_goal.jpg", args.image_size, device)

        # now move it back to the start position to get the current image
        robot.pos = start

        # let's run the model for max_steps or until the goal is reached whichever is first
        max_steps = 50
        for i in range(max_steps):
            # get the current image
            print("pos:", robot.pos)
            robot.render()
            robot.fig.savefig(f"/tmp/cpt/visual_nav2d_obs.jpg")
            curr_image = get_image_tensor("/tmp/cpt/visual_nav2d_obs.jpg", args.image_size, device)

            # get latent actions from the ILPO model
            _, zt, zmu, zsigma = model(curr_image, goal_image)
            # get the action from the action decoder
            action = action_decoder(zt)

            # squeeze action to remove batch dimension and move it to the cpu
            action = action.squeeze(0)
            action = action.detach().cpu().numpy()

            if args.smooth_action:
                action_buffer = np.vstack([action_buffer[1:, :], np.zeros((1, 2))])
                action_buffer = 0.1 * action_buffer + 0.9 * action
                action_final = action_buffer[0]
            else:
                action_final = action[0]

            robot.step(action_final)

            print("action:", action_final)
            print("goal:", robot.goal)

            # print("pred_obs:", next_obs_pred[0].detach().cpu().numpy())
            # print("true_obs:", obs)

            if np.linalg.norm(robot.pos - goal) < 0.1:
                print("Goal reached!")
                break
                # No need to call reset() because robot.step() will call it


if __name__ == "__main__":

    parser = get_arg_parser()

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, action_decoder, model_args = load_model(args, device)

    evaluation(model, action_decoder, args, model_args)
