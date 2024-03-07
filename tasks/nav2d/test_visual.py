import argparse
import colorsys
import os
import random
import sys
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import skimage.io
import torch
import torch.nn as nn
import torchvision
import visual.utils
import visual.vision_transformer as vits
from matplotlib.patches import Polygon
from PIL import Image
from skimage.measure import find_contours
from torchvision import transforms as pth_transforms
from visual.ilpo import MLP, ActionDecoder, Dynamics, ILPOWrapper, Policy


def test_visual_nav2d(model):

    import pudb

    pudb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
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
    args = parser.parse_args()
    assert os.path.isfile(args.pretrained_weights), "No pretrained weights found or file invalid"

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")

    model_args = state_dict["args"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    transformer = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

    embed_dim = transformer.embed_dim
    head = vits.DINOHead(embed_dim, model_args.out_dim, model_args.use_bn_in_head)

    model = ILPOWrapper(
        transformer,
        head,
        embed_dim=embed_dim,
        latent_action_dim=model_args.latent_action_dim,
        units=model_args.policy,
    )

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict["student"], strict=False)
    action_decoder = ActionDecoder(
        model_args.latent_action_dim, units=model_args.action_decoder, action_shape=(6, 2)
    )
    action_decoder.load_state_dict(state_dict["action_decoder"], strict=False)

    for p in model.parameters():
        p.requires_grad = False
    for p in action_decoder.parameters():
        p.requires_grad = False
    model.eval()
    action_decoder.eval()
    model.to(device)
    action_decoder.to(device)

    curr_file = "/home/gagan/Home/VideoIL/data/nav2d_visual/0/000000.jpg"
    goal_file = "/home/gagan/Home/VideoIL/data/nav2d_visual/0/000004.jpg"

    def get_image_tensor(file):
        image = Image.open(file)
        image = image.resize(args.image_size)
        image = pth_transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device)
        return image

    ot = get_image_tensor(curr_file)
    og = get_image_tensor(goal_file)

    out, zt, zmu, zsigma = model(ot, og)

    action = action_decoder(zt)

    zt = zt.detach().cpu().numpy()

    print("latent action", zt)
    print("action", action)
