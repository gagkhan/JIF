from common.mlp import MLP
import torch
from torch import nn
import torch.nn.functional as F


class ActionDecoder(nn.Module):
    def __init__(self, latent_action_dim, units=[64, 64], action_shape=None) -> None:
        self.latent_action_dim = latent_action_dim
        self.units = units
        super(ActionDecoder, self).__init__()
        self.action_shape = action_shape
        self.action_decoder_out_dim = action_shape[0] * action_shape[1]
        self.mlp = MLP(latent_action_dim, self.action_decoder_out_dim, units)

    def forward(self, x):
        out = self.mlp(x)
        out = out.view((-1, *self.action_shape))
        return out


def action_loss(actions_pred, actions):
    action_dim = actions.shape[2]
    assert(action_dim == 8)

    closs = cartesian_loss(actions_pred[:,:,[0,1,2,7]], actions[:,:,[0,1,2,7]])
    qloss = quaternion_loss(actions_pred[:,:,3:7], actions[:,:,3:7])
    loss = closs + qloss
    return loss

def cartesian_loss(cart_pred, cart):
    return F.mse_loss(cart_pred, cart)

def quaternion_loss(quat_pred, quat):
    '''
    This loss is from https://stackoverflow.com/questions/73380197
    '''
    quat_pred = F.normalize(quat_pred, dim=-1)
    quat      = F.normalize(quat,      dim=-1)
    q_diff = quat_difference(quat_pred, quat)
    # Flip signs, so w>0
    q_diff[q_diff[...,-1]<0] *= -1
    # Optimal w=1
    q_diff[...,3] -= 1
    # Loss
    return F.mse_loss(q_diff, torch.zeros_like(q_diff))

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
    x0, y0, z0, w0 = q0.chunk(4, dim=-1)
    x1, y1, z1, w1 = q1.chunk(4, dim=-1)
    x = w0*x1 + x0*w1 + y0*z1 - z0*y1
    y = w0*y1 - x0*z1 + y0*w1 + z0*x1
    z = w0*z1 + x0*y1 - y0*x1 + z0*w1
    w = w0*w1 - x0*x1 - y0*y1 - z0*z1
    return torch.cat([x,y,z,w], dim=-1)
def quat_difference(_from, _to):
    '''
    Return diff=inv(_from)*_to, such that _to=_from*diff
    '''
    q0 = quat_inverse(_from)
    q1 = _to
    return quat_multiply(q0, q1)
