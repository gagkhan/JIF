import os
import numpy as np


def quat_inverse(q):
    '''
    Return inverse of q
    '''
    q_inv = q
    q_inv[...,0:3] = -q_inv[...,0:3]
    return q_inv
def quat_multiply(q0, q1):
    '''
    Return multiplication of q0 and q1
    '''
    x0, y0, z0, w0 = np.split(q0, 4, axis=-1)
    x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
    x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    z = w0*z1 + x0*y1 - y0*x1 + z0*w1
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return np.concatenate([x,y,z,w], axis=-1)
def quat_difference(_from, _to):
    '''
    Return diff=inv(_from)*_to, such that _to=_from*diff
    '''
    q0 = quat_inverse(_from)
    q1 = _to
    return quat_multiply(q0, q1)


if __name__ == "__main__":

    src_dir = '/ssd01/gagan/cpt_data/ours/12_02_pickrobot'

    # List all subdirectories in the source directory
    subdirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    # Sort
    subdirs = sorted(subdirs, key=lambda x: int(x.split('_')[1]))

    # Fix actions
    for dir in subdirs:
        commands  = np.load(os.path.join(src_dir, dir, "commands.npy"))
        ee_states = np.load(os.path.join(src_dir, dir, "ee_states.npy"))
        actions   = np.empty_like(commands)
        actions[:,[0,1,2,7]] = commands[:,[0,1,2,7]] - ee_states[:,[0,1,2,7]]
        actions[:,[3,4,5,6]] = quat_difference(ee_states[:,[3,4,5,6]], commands[:,[3,4,5,6]])

        print(dir)
        # np.save(os.path.join(src_dir, dir, "actions.npy"), actions)