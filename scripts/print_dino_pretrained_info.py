import argparse
import os

import torch
from termcolor import colored, cprint


def main(args):

    if os.path.exists(args.path):
        print("path exists! loading the model")
        state_dict = torch.load(args.path, map_location="cpu")

        cprint("The state dict consists of student, teacher and early_stopping_epoch keys")
        print([key for key in state_dict.keys()])

        # print([key for key in state_dict["student"].keys()])

        cprint(
            "This implies the student weights are located at state_dict['student'].module.backbone",
            "green",
        )
        cprint(
            "...and the student head weights are located at state_dict['student'].module.head",
            "green",
        )

        # print([key for key in state_dict["teacher"].keys()])

        cprint(
            "This implies the teacher weights are located at state_dict['teacher'].backbone",
            "green",
        )
        cprint("...and the teacher head weights are located at state_dict['teacher'].head", "green")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)

    args = parser.parse_args()

    main(args)
