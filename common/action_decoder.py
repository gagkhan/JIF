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


def action_loss(actions, actions_pred, mask):
    action_dim = actions.shape[2]
    assert(action_dim == 8 or action_dim == 1)

    # If action_dim == 1, assume action loss is not used
    if action_dim == 1:
        return torch.tensor(0.0)

    closs = cartesian_loss(actions_pred[:,:,[0,1,2,7]], actions[:,:,[0,1,2,7]], mask[:,:,[0,1,2,7]])
    qloss = quaternion_loss(actions_pred[:,:,3:7], actions[:,:,3:7], mask[:,:,3:7])
    loss = closs + qloss
    return loss

def cartesian_loss(actions, actions_pred, mask):

    error = mask * (actions_pred - actions)
    sqerror = error * error
    aloss = (sqerror).mean()

    return aloss

# def quaternion_loss(quat_pred, quat, mask):
#     '''
#     This loss is from https://datascience.stackexchange.com/questions/36370
#     '''
#     quat_pred = F.normalize(quat_pred, dim=-1)
#     quat      = F.normalize(quat,      dim=-1)
#     loss = (1 - abs((quat_pred * quat).sum(dim=2))) * mask[:,:,0] # (batch_size, action_chunk_len)
#     print(loss.shape)
#     loss = loss.mean()
#     return loss

def quaternion_loss(quat_pred, quat, mask):
    '''
    This loss is from https://stackoverflow.com/questions/73380197
    '''
    def quat_inverse(q):
        '''
        Return inverse of q
        q: quaternion tensor of shape (batch_size, chunk_len, 4), assumed normalized
        '''
        q_inv = q
        q_inv[:,:,0:3] = -q_inv[:,:,0:3]
        return q_inv
    def quat_multiply(q0, q1):
        '''
        Return multiplication of q0 and q1
        q0,q1: quaternion tensor of shape (batch_size, chunk_len, 4), assumed normalized
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
        Return _from^-1 * _to, (think of it as _to/_from)
        _from,_to: quaternion tensor of shape (batch_size, chunk_len, 4), assumed normalized
        '''
        q0 = quat_inverse(_from)
        q1 = _to
        return quat_multiply(q0, q1)
    B,C,D = quat_pred.shape
    quat_pred = F.normalize(quat_pred, dim=-1)
    quat      = F.normalize(quat,      dim=-1)
    q_diff = quat_difference(quat_pred, quat)
    q_diff = q_diff.view(B*C,D)
    # Flip signs, so w>0
    q_diff_flipped = torch.tensor(q_diff)
    for i, q in enumerate(q_diff):
        if q[3] < 0:
            q_diff_flipped[i] = -q
    # Optimal w=1
    q_diff_flipped[:,3] -= 1
    q_diff_flipped = q_diff_flipped.view(B,C,D)
    # Loss
    error = q_diff_flipped * mask
    sqerror = error * error
    loss = sqerror.mean()
    return loss