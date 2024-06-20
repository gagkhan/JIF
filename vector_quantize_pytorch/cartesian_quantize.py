import numpy as np
import torch
from torch import nn


class CartesianActionChunkQuantize:

    num_lookup = {7: (5, 3), 13: (6, 4), 21: (7, 5), 31: (8, 6)}

    def __init__(self, num_actions=13, action_scale=0.0008):
        # initialize some parameters
        # number of discrete actions
        # codebook : (n, () )

        assert num_actions in self.num_lookup.keys(), "num_actions=f{num_actions} not supported"

        num_phi, num_theta = self.num_lookup[num_actions]

        theta = np.linspace(0, np.pi, num=num_theta, endpoint=True)
        phi = np.linspace(0, 2 * np.pi, num=num_phi, endpoint=True)

        v = np.zeros((len(theta) * len(phi), 3))
        i = 0
        for t in theta:
            for p in phi:
                v[i] = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
                i += 1
        v[np.abs(v) < 1e-8] = 0
        weights = np.unique(v, axis=0)
        weights = np.vstack([weights, np.array([0, 0, 0])])
        weights = torch.from_numpy(weights).to(dtype=torch.float32)
        self.weights = weights

        # self.codebook = nn.Embedding(len(weights), 3)
        # for param in self.codebook.parameters():
        # param.requires_grad = False
        # self.codebook.weight[:] = weights
        self.action_scale = action_scale

    # @torch.no_grad
    def __call__(self, x):
        """
        Args:
            x: batch_size, chunk_size, action_dim
        Returns:
          _, idx, _ .. values to be consistent with interface in vector_quantize_pytorch/VectorQuantize

        """

        chunk_size = x.shape[1]
        x = torch.sum(x, dim=1)
        weights = self.weights * self.action_scale * chunk_size
        idx = torch.argmin(torch.cdist(weights, x.unsqueeze(0)), dim=1).view(-1)
        # idx = idx.view(-1, 1)
        return None, idx, None


def test_cartesian_action_chunk_quantize():

    quanitzer = CartesianActionChunkQuantize()

    # create dummpy action chunk
    x = torch.zeros(16, 6, 3)
    for i in range(x.shape[0]):
        r = torch.rand((3,)) - 0.5
        r /= torch.norm(r)
        x[i] = 0.0008 * r.repeat((6, 1))

    # print(x.shape)
    # print(x)

    _, idx, _ = quanitzer(x)

    print(idx)


if __name__ == "__main__":

    test_cartesian_action_chunk_quantize()
