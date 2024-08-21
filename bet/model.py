import torch
import torch.nn.functional as F
from bet.gpt import GPT
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, units):
        super().__init__()
        layers = []
        for outsize in units:
            layers.append(nn.Linear(input_size, outsize))
            layers.append(nn.GELU())
            input_size = outsize
        layers.append(nn.Linear(input_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


''' LatentPolicy networks '''

class DebugMLP(nn.Module):
    def __init__(
        self,
        input_img_dim,
        seq_len,
        n_embd,
        units,
    ):
        super().__init__()
        input_dim = (seq_len + 1) * input_img_dim
        self.n_embd = n_embd

        self.debug_mlp = MLP(input_dim, n_embd, units)

    def forward(self, img_seq, goal_img):
        """
        Args:
            img_seq:  (B, seq_len, input_img_dim)
            goal_img: (B,       1, input_img_dim)
        Returns:
            x:        (B,       1, n_embd)
        """
        # Reshape each input to (B, -1)
        img_seq = img_seq.flatten(start_dim=1)
        goal_img = goal_img.flatten(start_dim=1)

        # Concatenate
        x = torch.cat([img_seq, goal_img], dim=1)

        # Forward
        x = self.debug_mlp(x).unsqueeze(1)
        return x


class BeT(nn.Module):

    def __init__(
        self,
        input_img_dim,
        seq_len,
        n_layer=4,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=True,
        causal=False,
    ):
        super().__init__()
        self.n_embd = n_embd

        self.gpt = GPT(n_layer, n_head, n_embd, seq_len+1, bias, dropout, causal)
        self.proj_img = nn.Linear(input_img_dim, n_embd)

    def forward(self, img_seq, goal_img):
        """
        Args:
            img_seq:  (B, seq_len, input_img_dim)
            goal_img: (B,       1, input_img_dim)
        Returns:
            x:        (B, seq_len+1, n_embd)
        """
        # Project each input to (B, -1, n_embed)
        B, T, *O = img_seq.shape
        proj_img_seq  = self.proj_img(img_seq .view(B * T, *O)).view(B, T, self.n_embd)

        B, T, *O = goal_img.shape
        proj_goal_img = self.proj_img(goal_img.view(B * T, *O)).view(B, T, self.n_embd)

        # Concatenate
        x = torch.cat([proj_img_seq, proj_goal_img], dim=1)

        # Forward
        x = self.gpt(x)
        return x


''' ActionDecoder network '''

class ActionDecoder(nn.Module):
    '''
    Args:
        action_dim:        Dimension of the action (3)
        action_chunk_len:  Number of actions in an action chunk
        num_quantizers:    Number of quantizer layers in rvq layer
        num_actions:       Number of quantized encoded action chunk, i.e. codebook_size
        n_embd:            BET outputs (B, seq_len+1, n_embd)
        use_ee:            Whether to use end effector positions
    '''

    def __init__(
        self,
        action_dim,
        action_chunk_len,
        num_quantizers,
        num_actions,
        n_embd=128,
        use_ee=False,
    ):
        super().__init__()
        self.G = num_quantizers
        self.C = num_actions
        self.W = action_chunk_len
        self.A = action_dim
        act1_input_dim = n_embd + (3 if use_ee else 0)
        act2_input_dim = act1_input_dim + self.C
        off_output_dim = self.G * self.C * self.W * self.A

        # Layers to predict quantizer indices
        self.act1 = MLP(act1_input_dim, self.C, units=[64, 64])
        self.act2 = MLP(act2_input_dim, self.C, units=[64, 64])
        # Layer to predict offsets (DECIDE WHETHER TO USE THIS LATER)
        self.off  = MLP(act1_input_dim, off_output_dim, units=[64, 64])

    def forward(self, x, ee=None):
        '''
        Args:
            x:  output of LatentPolicy network;  (B, seq_len+1, n_embd)
            ee: end effector position sequences; (B, seq_len, 3)
        Returns:
            logits1: for quantizer layer 1;      (B, num_actions)
            index1:  for quantizer layer 1;      (B,)
            logits2: for quantizer layer 2;      (B, num_actions)
            index2:  for quantizer layer 2;      (B,)
            offsets: for quantized action chunk; (B, action_chunk_len, action_dim)
        '''
        # act1 to predict first quantizer layer index
        x  = x [:, -1]
        act1_input = x
        if ee is not None:
            ee = ee[:, -1]
            act1_input = torch.cat([act1_input, ee], dim=1)

        logits1 = self.act1(act1_input)
        index1  = self.act_softmax(logits1)
        onehot1 = F.one_hot(index1, num_classes=self.C)

        # act2 to predict second quantizer layer index
        act2_input = torch.cat([act1_input, onehot1], dim=1)

        logits2 = self.act2(act2_input)
        index2  = self.act_softmax(logits2)
        onehot2 = F.one_hot(index2, num_classes=self.C)

        # off to predict offsets
        off_input = act1_input

        offsets = self.off(off_input).view(-1, self.G, self.C, self.W, self.A)
        offsets = [(o[0, i1] + o[1, i2]) for (o, i1, i2) in zip(offsets, index1, index2)]
        offsets = torch.stack(offsets)
        
        # Compile return variable
        ret = {
            "logits1": logits1,
            "index1" : index1,

            "logits2": logits2,
            "index2" : index2,
            
            "offsets" : offsets,
        }

        return ret

    @torch.no_grad()
    def act_softmax(self, logits):
        p = F.softmax(logits, dim=-1)
        pred_indices = torch.multinomial(p, num_samples=1).squeeze(1) # (batch_size)
        return pred_indices

    @torch.no_grad()
    def act_top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ 
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        
        Code taken from https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
        """
        assert logits.dim() == 2  # (batch_size, num_actions)
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Replace logits to be removed with -inf in the sorted_logits
        sorted_logits[sorted_indices_to_remove] = filter_value
        # Then reverse the sorting process by mapping back sorted_logits to their original position
        logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
        
        pred_indices = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1) # (batch_size)
        return pred_indices


''' LatentPolicy network builders '''
# network builders BeT for 'base' and 'large' sizes based on the sizes used in https://arxiv.org/pdf/2206.11251
# 'base' and 'large' are based on sizes used for push block and kitchen tasks respectively

def bet_base(input_img_dim, seq_len, causal):

    model = BeT(
        input_img_dim=input_img_dim,
        seq_len=seq_len,
        n_layer=4,
        n_head=4,
        n_embd=72,
        dropout=0.1,
        bias=False,
        causal=causal,
    )

    return model


def bet_large(input_img_dim, seq_len, causal):

    model = BeT(
        input_img_dim=input_img_dim,
        seq_len=seq_len,
        n_layer=6,
        n_head=6,
        n_embd=120,
        dropout=0.1,
        bias=False,
        causal=causal,
    )

    return model


def mlp_large(input_img_dim, seq_len, causal=False):

    model = DebugMLP(
        input_img_dim=input_img_dim,
        seq_len=seq_len,
        n_embd=120,
        units=[512, 512],
    )

    return model


def mlp_base(input_img_dim, seq_len, causal=False):

    model = DebugMLP(
        input_img_dim=input_img_dim,
        seq_len=seq_len,
        n_embd=72,
        units=[64, 64],
    )

    return model


def mlp_small(input_img_dim, seq_len, causal=False):

    model = DebugMLP(
        input_img_dim=input_img_dim,
        seq_len=seq_len,
        n_embd=36,
        units=[16, 16],
    )

    return model


''' Testing '''

def test_reshaping():

    n_embd = 16
    context_len = 4
    batch_size = 8

    x = torch.rand((batch_size, context_len, n_embd))
    B, T, *O = x.shape
    x = x.view(B * T, *O)
    assert x.shape[0] == B * T
    x = x.view(B, T, *O)
    assert x.shape[0] == B and x.shape[1] == T


def behavior_transformer(causal=False):

    input_img_dim = 16
    seq_len = 3
    num_actions = 10
    n_layer = 4
    n_head = 2
    n_embd = 128
    dropout = 0.0
    bias = True
    batch_size = 8

    model = BeT(
        input_img_dim=input_img_dim,
        seq_len=seq_len,
        num_actions=num_actions,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        causal=causal,
    )

    # test forward pass with BeT
    x = torch.rand((batch_size, seq_len+1, input_img_dim))
    out = model(x)
    assert out.shape[0] == batch_size
    assert out.shape[1] == num_actions
    assert len(torch.nonzero(out, as_tuple=True)) > 0

    # create one hot action vectors
    actions = torch.zeros((batch_size, num_actions))
    actions[torch.arange(batch_size), torch.randint(0, num_actions, (batch_size,))] = 1

    print("loss: ", model.loss(x, actions))
    print("actions: ", model.act(x))


def test_behavior_transformer_causal():
    behavior_transformer(causal=True)


def test_behavior_transformer():
    behavior_transformer(causal=False)


if __name__ == "__main__":

    test_behavior_transformer()
