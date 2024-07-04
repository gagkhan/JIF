import os
import numpy as np
import matplotlib.pyplot as plt


DEPTH_DATA_ROOT = "/home/gagan/Home/VideoIL/data/ours/ours_tabletop/test_depth/demo_1"
FILE = "depth_000000.png"

import cv2


if __name__ == "__main__":

    # cv2.imshow("Depth Image",)

    depth = cv2.imread(os.path.join(DEPTH_DATA_ROOT, FILE), cv2.IMREAD_UNCHANGED) * 1.0 / 7000
    # Assume depth_image.png is 16 bits grayscale.
    # depth = cv2.resize(depth, interpolation=cv2.INTER_NEAREST)

    depth = depth * 255
    depth = depth.astype(np.uint8)

    # cv2.imshow("Depth Image", depth)

    plt.imshow(depth, cmap="gray", vmin=0, vmax=255)
    plt.show()

    input("Press Enter to continue...")
