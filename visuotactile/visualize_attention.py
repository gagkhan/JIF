import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import visual.utils as utils
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from visuotactile.utils import build_vitact_encoder


def get_num(name):
    match = re.search(r"\d{1,}", name)
    num = -1
    if match:
        num = int(match.group())
    return num


def get_args_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="/home/gagan/Home/VideoIL/data/ours/aug09_pickhuman",
        type=str,
        help="Please specify path to demonstrations",
    )
    parser.add_argument("--output_dir", default="./debug")
    parser.add_argument("--demo_num", default=1, type=int, help="Demo number to be visualized")
    parser.add_argument(
        "--use_cam2",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not wrist view camera (cam3) is used.""",
    )
    parser.add_argument(
        "--use_cam3",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not wrist view camera (cam3) is used.""",
    )

    # ViTacT
    parser.add_argument(
        "--encoder_arch",
        default="vitact_small",
        type=str,
        choices=["vitact_tiny", "vitact_small", "vitact_base"],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load before training.",
    )

    # Attention
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )

    return parser


def save_attn_map(fn, attentions, base_img, w_featmap, h_featmap):

    nh = attentions.shape[0]
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0),
            scale_factor=args.patch_size,
            mode="nearest",
        )[0]
        .cpu()
        .numpy()
    )

    plt.imsave(
        fname=fn,
        arr=sum(attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0])),
        cmap="inferno",
        format="jpg",
    )
    heatmap = np.array(Image.open(fn))
    attn_img = cv2.addWeighted(heatmap, 0.5, np.array(base_img), 0.5, 0)
    cv2.imwrite(fn, attn_img)


def read_and_adjust(fn, args):
    img = Image.open(fn)
    img = np.array(img)
    # make the image divisible by the patch size
    w, h = (
        img.shape[0] - img.shape[0] % args.patch_size,
        img.shape[1] - img.shape[1] % args.patch_size,
    )
    img = img[:w, :h, :]
    w_featmap = img.shape[0] // args.patch_size
    h_featmap = img.shape[1] // args.patch_size

    img = Image.fromarray(img)

    return img, w_featmap, h_featmap


def main(args):

    args.data_path
    demodir = os.path.join(args.data_path, f"demo_{args.demo_num}")
    tactile_path = os.path.join(args.data_path, f"demo_{args.demo_num}/tactile.npy")
    tactile_data = np.load(tactile_path)
    demolen = len(tactile_data)
    cam1_path = os.path.join(demodir, f"cam1/color")
    cam2_path = os.path.join(demodir, f"cam2/color")
    cam3_path = os.path.join(demodir, f"cam3/color")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    teacher, embed_dim = build_vitact_encoder(args)
    teacher = teacher.cuda()

    for i in tqdm(range(demolen)):
        frame_no = str(i).zfill(6)

        # read images as PIL
        img1_path = os.path.join(cam1_path, f"color_{frame_no}.png")
        img1_base, w1, h1 = read_and_adjust(img1_path, args)
        img1 = transform(img1_base).cuda().unsqueeze(0)
        tactile = torch.tensor(tactile_data[i], dtype=torch.float32).cuda().unsqueeze(0)
        x = [img1, tactile]
        if args.use_cam2:
            img2_path = os.path.join(cam2_path, f"color_{frame_no}.png")
            img2_base, w2, h2 = read_and_adjust(img2_path, args)
            img2 = transform(img2_base).cuda().unsqueeze(0)
            x.append(img2)

        if args.use_cam3:
            img3_path = os.path.join(cam3_path, f"color_{frame_no}.png")
            img3_base, w3, h3 = read_and_adjust(img2_path, args)
            img3 = transform(img3_base).cuda().unsqueeze(0)
            x.append(img3)

        # x = [img1, tactile, img2, img3]
        attentions = teacher.get_last_selfattention(x).detach()

        nh = attentions.shape[1]  # number of head
        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # saving attention for first view only
        k = 0
        attention = attentions[:, k : k + w1 * h1]
        save_attn_map(os.path.join(args.output_dir, f"cam1_attn_{i}.jpg"), attention, img1_base, w1, h1)
        k += w1 * h1
        if args.use_cam2:
            # saving attention for first view only
            attention = attentions[:, k : k + w2 * h2]
            k += w2 * h2
            save_attn_map(os.path.join(args.output_dir, f"cam2_attn_{i}.jpg"), attention, img2_base, w2, h2)

        if args.use_cam3:
            # saving attention for first view only
            attention = attentions[:, k : k + w3 * h3]
            save_attn_map(os.path.join(args.output_dir, f"cam3_attn_{i}.jpg"), attention, img3_base, w3, h3)


def make_video(args):

    args.output_dir


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
