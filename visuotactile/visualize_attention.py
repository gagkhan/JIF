import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import visual.utils as utils
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip
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

    # ViTacT
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load before training.",
    )
    parser.add_argument(
        "--freeze_encoder",
        type=utils.bool_flag,
        default=True,
        help=""" Whether to freeze encoder (required to create the encoder with network builder)""",
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


def save_attn_map(fn, nh, attentions, base_img, w_featmap, h_featmap, patch_size):

    # nh = attentions.shape[0]
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0),
            scale_factor=patch_size,
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

    training_args = torch.load(args.pretrained_weights, map_location="cpu")["args"]
    args.encoder_arch = training_args.encoder_arch
    args.patch_size   = training_args.patch_size
    args.use_tactile  = training_args.use_tactile
    args.use_cam2     = training_args.use_cam2
    args.use_cam3     = training_args.use_cam3

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

    teacher, embed_dim = build_vitact_encoder(args, "student")
    teacher = teacher.cuda()
    teacher.eval()

    for i in tqdm(range(demolen)):
        frame_no = str(i).zfill(6)

        # read images as PIL
        img1_path = os.path.join(cam1_path, f"color_{frame_no}.png")
        img1_base, w1, h1 = read_and_adjust(img1_path, args)
        img1 = transform(img1_base).cuda().unsqueeze(0)
        tactile = torch.tensor(tactile_data[i], dtype=torch.float32).cuda().unsqueeze(0)
        x = [img1]
        if args.use_tactile:
            x.append(tactile)
        if args.use_cam2:
            img2_path = os.path.join(cam2_path, f"color_{frame_no}.png")
            img2_base, w2, h2 = read_and_adjust(img2_path, args)
            img2 = transform(img2_base).cuda().unsqueeze(0)
            x.append(img2)
        if args.use_cam3:
            img3_path = os.path.join(cam3_path, f"color_{frame_no}.png")
            img3_base, w3, h3 = read_and_adjust(img3_path, args)
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
        save_attn_map(os.path.join(args.output_dir, f"cam1_attn_{i}.jpg"), nh, attention, img1_base, w1, h1, args.patch_size)
        k += w1 * h1
        if args.use_cam2:
            # saving attention for first view only
            attention = attentions[:, k : k + w2 * h2]
            k += w2 * h2
            save_attn_map(os.path.join(args.output_dir, f"cam2_attn_{i}.jpg"), nh, attention, img2_base, w2, h2, args.patch_size)
        if args.use_cam3:
            # saving attention for first view only
            attention = attentions[:, k : k + w3 * h3]
            save_attn_map(os.path.join(args.output_dir, f"cam3_attn_{i}.jpg"), nh, attention, img3_base, w3, h3, args.patch_size)

    make_video(args, demolen)


def make_video(args, demolen):

    frames = []

    for i in range(demolen):

        cam1_file = os.path.join(args.output_dir, f"cam1_attn_{i}.jpg")
        cam2_file = os.path.join(args.output_dir, f"cam2_attn_{i}.jpg")
        cam3_file = os.path.join(args.output_dir, f"cam3_attn_{i}.jpg")

        im1 = Image.open(cam1_file)
        im2 = Image.open(cam3_file)
        im3 = Image.open(cam2_file)

        w = im1.size[0] + im2.size[0] + im3.size[0]
        h = max(im1.size[1], im2.size[1], im3.size[1])
        im = Image.new("RGB", (w, h))

        im.paste(im1)
        im.paste(im2, (im1.size[0], 0))
        im.paste(im3, (im1.size[0] + im2.size[0], 0))

        frames.append(im)

    fps = 4  # Frames per second
    clip = ImageSequenceClip([np.array(image) for image in frames], fps=fps)
    clip.write_videofile(os.path.join(args.output_dir, "video.mp4"), codec="libx264")


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
