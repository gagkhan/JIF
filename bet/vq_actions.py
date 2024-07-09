from torch import nn, Tensor

from bet.model import MLP
from vector_quantize_pytorch import VectorQuantize


class ActionQuantizer(nn.Module):
    '''
    This class quantizes the continuous actions into discrete actions.

    action_dim:        Dimension of the action (3)
    action_chunk_size: Number of actions in an action chunk
    embedding_dim:     Dimension of the encoded action chunk 
    num_embeddings:    Number of quantized encoded action chunk
    '''
    def __init__(
        self,
        action_dim, 
        action_chunk_size, 
        embedding_dim,
        num_embeddings
    ) -> None:

        super().__init__()
        flat_input_dim = action_dim * action_chunk_size
        
        self.encoder   = MLP(flat_input_dim, embedding_dim, [])
        self.quantizer = VectorQuantize(embedding_dim, num_embeddings)
        self.decoder   = MLP(embedding_dim, flat_input_dim, [])

    def forward(self, x: Tensor):              # (batch, action_chunk_size, action_dim)
        x_flat =  x.flatten(start_dim=1)       # (batch, action_chunk_size*action_dim)

        z_e          = self.encoder(x_flat)    # (batch, embedding_dim)
        z_q, loss, _ = self.quantizer(z_e)     # (batch, embedding_dim)
        x_recon      = self.decoder(z_q)       # (batch, action_chunk_size*action_dim)

        x_recon_unflat = x_recon.view(x.shape) # (batch, action_chunk_size, action_dim)
        return x_recon_unflat, loss