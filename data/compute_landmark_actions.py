import os
import pickle

import numpy as np


def get_array(joints, hand="Left"):
    poses = []

    # find a good default value for lm_pos
    for frame in range(len(joints[hand])):
        if len(joints[hand][frame]) == 21:
            lm_pos_prev = joints[hand][frame]

    # for each frame, get the 3D coordinates of each joint
    for frame in range(len(joints[hand])):  # left and right should have the same number of frames
        pos = []
        if len(joints[hand][frame]) != 21:
            lm_pos = lm_pos_prev
        else:
            lm_pos = joints[hand][frame]
        # print(len(lm_pos))
        for i in range(21):
            pos.extend([lm_pos[i]["X"], lm_pos[i]["Y"], lm_pos[i]["Z"]])
        poses.append(pos)
    return np.array(poses)


if __name__ == "__main__":

    src_path = "/home/sarahp/ours_v2_frames/"
    dest_path = "/home/gagan/Home/VideoIL/data/ours/ours_v2_frames/"

    for demo in os.listdir(src_path):
        file = os.path.join(src_path, demo, "joint_angles.pkl")
        if os.path.exists(file) is False:
            continue
        # print(demo)
        data = pickle.load(open(file, "rb"))
        left_poses = get_array(joints=data, hand="Left")
        right_poses = get_array(joints=data, hand="Right")
        poses = np.concatenate((left_poses, right_poses), axis=1)
        actions = poses[1:, :] - poses[:-1, :]
        act_path = os.path.join(dest_path, demo, "actions.npy")
        np.save(act_path, actions)
