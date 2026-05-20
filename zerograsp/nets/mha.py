import torch as th
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding


# Move head forward and fold into batch dim. dimensions become (B, nh, S, hs)
def _split_heads(t: th.Tensor, B: int, S: int, H: int, Hs: int):
    return t.view(B, S, H, Hs)


class MHA(pl.LightningModule):

    def __init__(self, config, attn_type='self') -> None:
        super(MHA, self).__init__()
        self.num_heads = config.num_heads    # 4
        self.dim_mae = config.dim_mae    # 128
        self.head_dim = self.dim_mae // self.num_heads
        self.max_freq = config.max_freq
        self.pe_type = config.pe_type
        self.attn_type = attn_type
        if attn_type == 'self':
            self.qkv_proj = nn.Linear(self.dim_mae, 3 * self.dim_mae, bias=True)
        else:
            self.q_proj = nn.Linear(self.dim_mae, self.dim_mae, bias=True)
            self.kv_proj = nn.Linear(self.dim_mae, 2 * self.dim_mae, bias=True)
        self.resid_drop = nn.Dropout(config.resid_dropout, inplace=False)
        self.proj = nn.Linear(self.dim_mae, self.dim_mae, bias=True)
        self.rotary_embeddings = RotaryEmbedding(dim=self.head_dim // 3, freqs_for='pixel', max_freq=self.max_freq)

    def rotary_positional_encode(self, x, voxel_index):
        """
        x: (B, N, Hs, D)
        voxel_index: (B, N, 3)
        """
        freqs_x = self.rotary_embeddings(voxel_index[:, :, 0])
        freqs_y = self.rotary_embeddings(voxel_index[:, :, 1])
        freqs_z = self.rotary_embeddings(voxel_index[:, :, 2])
        freqs = th.cat((freqs_x, freqs_y, freqs_z), dim=-1)
        freqs = freqs.unsqueeze(2).repeat(1, 1, self.num_heads, 1)
        return apply_rotary_emb(freqs, x)

    def _run_attention(self, q, k, v):
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.resid_drop.p if self.training else 0.0,
        )
        return y.permute(0, 2, 1, 3)

    def _run_segmented_attention(self, q, k, v, attn_lengths):
        outputs = []
        start = 0
        for length in attn_lengths:
            end = start + int(length)
            outputs.append(self._run_attention(q[:, start:end], k[:, start:end], v[:, start:end]))
            start = end

        if start != q.shape[1]:
            raise ValueError("Segmented attention lengths do not cover the full sequence.")

        return th.cat(outputs, dim=1)

    def forward(self, x, y=None, x_index=None, y_index=None, attn_lengths=None):

        B, S_q, _ = x.size()

        if self.attn_type =='self':
            qkv = self.qkv_proj(x)
            qkv = _split_heads(qkv, B, S_q, self.num_heads, 3 * self.head_dim)
            q, k, v = qkv.chunk(3, dim=-1)
            y_index = x_index
        else:
            S_kv = y.shape[1]
            q = self.q_proj(x)
            kv = self.kv_proj(y)
            q = _split_heads(q, B, S_q, self.num_heads, self.head_dim)
            kv = _split_heads(kv, B, S_kv, self.num_heads, 2 * self.head_dim)
            k, v = kv.chunk(2, dim=-1)

        if self.pe_type == 'rope': # INFO(sh8): RoPE must be applied right before an attention map is computed
            q = self.rotary_positional_encode(q, x_index)
            k = self.rotary_positional_encode(k, y_index)

        if attn_lengths is not None and self.attn_type == 'self':
            y = self._run_segmented_attention(q, k, v, attn_lengths)
        else:
            y = self._run_attention(q, k, v)

        y = y.to(q.dtype)
        y = (y.view(B, S_q, self.num_heads, self.head_dim).flatten(start_dim=2, end_dim=3))
        y = self.resid_drop(self.proj(y))
        return y
