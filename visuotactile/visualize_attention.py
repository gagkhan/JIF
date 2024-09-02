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


def main(args):

    # get path to demos:
    # for each demo:
    # for each frame in demo

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
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    teacher, embed_dim = build_vitact_encoder(args)
    teacher = teacher.cuda()

    for i in range(demolen):
        frame_no = str(i).zfill(6)
        img_path = os.path.join(cam2_path, f"color_{frame_no}.png")

        img1_ = Image.open(os.path.join(cam1_path, f"color_{frame_no}.png"))
        img2 = Image.open(os.path.join(cam2_path, f"color_{frame_no}.png"))
        img3 = Image.open(os.path.join(cam3_path, f"color_{frame_no}.png"))

        tactile = torch.tensor(tactile_data[i], dtype=torch.float32).cuda().unsqueeze(0)

        img1 = transform(img1_).cuda().unsqueeze(0)
        img2 = transform(img2).cuda().unsqueeze(0)
        img3 = transform(img3).cuda().unsqueeze(0)

        x = [img1, tactile, img2, img3]
        # teacher(x)
        attentions = teacher.get_last_selfattention(x)

        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        print("before")
        print(th_attn[0].shape)

        # break

        th_attn = th_attn[:, :196]  # get the 196 positions corresponding to the first image

        w_featmap = 14
        h_featmap = 14

        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = (
            nn.functional.interpolate(
                th_attn.unsqueeze(0),
                scale_factor=args.patch_size,
                mode="nearest",
            )[0]
            .cpu()
            .numpy()
        )
        print("after:")
        print(th_attn.shape)

        plt.imsave(
            fname=f"debug/attn_{i}.jpg",
            arr=sum(th_attn[i] * 1 / th_attn.shape[0] for i in range(th_attn.shape[0])),
            cmap="inferno",
            format="jpg",
        )

        # heatmap = cv2.applyColorMap(th_attn, cv2.COLORMAP_JET)
        # print(heatmap.shape)
        img1_ = img1_.resize((224, 224), Image.BICUBIC)
        original_img = np.array(img1_)
        # print(original_img.shape)
        heatmap = np.array(Image.open(f"debug/attn_{i}.jpg"))
        attn_img = cv2.addWeighted(heatmap, 0.5, original_img, 0.5, 0)
        # cv2.imwrite(, heatmap)
        cv2.imwrite(f"debug/attn_overlay_{i}.jpg", attn_img)


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
