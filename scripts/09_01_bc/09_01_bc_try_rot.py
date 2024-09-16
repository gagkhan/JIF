import numpy as np
import os, sys


def quat_inverse(q):
    '''
    Return inverse of q
    q: quaternion tensor [[x,y,z,w]], of shape (batch_size, 4)
    '''
    inv_op = np.ones_like(q)
    inv_op[:,0:3] = -1
    q_inv = q * inv_op
    return q_inv

def quat_multiply(q0, q1):
    '''
    Return multiplication of q0 and q1
    q0,q1: quaternion tensor [[x,y,z,w]], of shape (batch_size, 4)
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
    Return _from^-1 * _to, (think of it as _to/_from)
    _from,_to: quaternion tensor [[x,y,z,w]], of shape (batch_size, 4)
    '''
    q0 = quat_inverse(_from)
    q1 = _to
    return quat_multiply(q0, q1)


task_dir_path = "/ssd01/gagan/cpt_data/ours/aug30_pickrobot_quat"
demo_dir_paths = [f.path for f in os.scandir(task_dir_path) if f.is_dir()]

for demo_dir_path in demo_dir_paths:
    # load files
    ee_states_path = os.path.join(demo_dir_path, "ee_states.npy")
    ee_states = np.load(ee_states_path)

    commands_path = os.path.join(demo_dir_path, "commands.npy")
    commands = np.load(commands_path)

    # calculate action
    xyzg = commands[:,[0,1,2,7]] - ee_states[:,[0,1,2,7]]
    quat = quat_difference(ee_states[:,3:7], commands[:,3:7])

    actions = np.empty_like(commands)
    actions[:,[0,1,2,7]] = xyzg
    actions[:,3:7]       = quat

    np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
    print('UPDATED ACTIONS!')

    # Save
    actions_path = os.path.join(demo_dir_path, "actions.npy")
    actions = np.load(actions_path)
    print(actions)
    # np.save(actions_path, actions)