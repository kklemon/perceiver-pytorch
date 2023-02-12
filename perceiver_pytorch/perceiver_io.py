import torch
import torch.nn.functional as F

from torch import nn, einsum
from einops import rearrange, repeat
from functools import wraps
from typing import Optional, Union, List


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    seq_len, rotate_dim = t.shape[-2], pos.shape[-1]
    pos = pos[..., -seq_len:, :]
    t, t_pass = t[..., :rotate_dim], t[..., rotate_dim:]
    t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return torch.cat((t, t_pass), dim=-1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, rotary_pos_emb=None, return_attn_weights=False):
        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)

        sim = einsum('b i d, b j d -> b i j', q, k)

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        out = self.to_out(out)

        if return_attn_weights:
            return out, rearrange(attn, '(b h) n d -> b h n d', h=h)

        return out


class LearnedScale(nn.Module):
    def __init__(self, init_scale: float = 0.0):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        return self.alpha * x


# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        rezero=False,
        seq_dropout_prob = 0.0,
        rotary_pos_emb = False,
        cross_attn_interval: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.latent_dim = latent_dim
        if num_latents:
            self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        else:
            self.latents = None

        norm_fn = lambda dim, fn, **kwargs: fn if rezero else PreNorm(dim, fn, **kwargs)
        scale_fn = LearnedScale if rezero else nn.Identity

        get_latent_attn = lambda: norm_fn(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: norm_fn(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(dim=max(32, latent_dim_head // 2))
        else:
            self.rotary_pos_emb = None

        self.self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])

        cache_args = {'_cache': weight_tie_layers}

        if cross_attn_interval is None:
            self.cross_attn_indices = [0]
        elif isinstance(cross_attn_interval, int):
            assert cross_attn_interval >= 1
            self.cross_attn_indices = list(range(0, depth, cross_attn_interval))
        elif isinstance(cross_attn_interval, list):
            assert max(cross_attn_interval) < depth and min(cross_attn_interval) >= 0
            self.cross_attn_indices = cross_attn_interval
        else:
            raise ValueError

        if isinstance(dim, (list, tuple)):
            assert len(dim) == len(self.cross_attn_indices)
            self.dims = dims = dim
        else:
            if self.cross_attn_layers
            self.dims = dims = [dim] * len(self.cross_attn_indices)

        for i in range(depth):
            if i in self.cross_attn_indices:
                dim, *dims = dims
                self.cross_attn_layers.append(nn.ModuleList([
                    norm_fn(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=dim),
                    norm_fn(latent_dim, FeedForward(latent_dim)),
                    scale_fn()
                ]))
            else:
                self.cross_attn_layers.append(None)

            self.self_attn_layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args),
                scale_fn()
            ]))

        self.decoder_cross_attn = norm_fn(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = norm_fn(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.decoder_scale = scale_fn()

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        queries = None,
        latents = None
    ):
        if isinstance(data, (tuple, list)):
            assert len(data) == len(self.cross_attn_indices)
            if mask is None:
                mask = [None] * len(self.cross_attn_indices)
            else:
                assert len(mask) == len(self.cross_attn_indices)

            b, *_, device = *data[0].shape, data[0].device
        else:
            data = [data] * len(self.cross_attn_indices)
            mask = [mask] * len(self.cross_attn_indices)

            b, *_, device = *data.shape, data.device

        if latents is not None:
            if latents.ndim == 2:
                latents = repeat(latents, 'n d -> b n d', b=b)
            if latents.ndim != 3 or latents.shape[0] != b or latents.shape[-1] != self.latent_dim:
                raise ValueError(f'Expected provided latents to have dimensions ({b}, n, {self.latent_dim}) or '
                                 f'(n, {self.latent_dim}) but found {tuple(latents.shape)}')
            x = latents
        else:
            if self.latents is None:
                raise ValueError('Module was initialized without learnable latents but no latents provided to '
                                 'forward()')
            x = repeat(self.latents, 'n d -> b n d', b = b)

        if self.rotary_pos_emb:
            rotary_pos_emb = self.rotary_pos_emb(max(t.shape[1] for t in data + [x]), device=device)
        else:
            rotary_pos_emb = None

        # structured dropout (as done in perceiver AR https://arxiv.org/abs/2202.07765)

        if self.training and self.seq_dropout_prob > 0.:
            if isinstance(data, (tuple, list)):
                data, mask = zip(*[dropout_seq(datum, maskum, self.seq_dropout_prob)
                                   for datum, maskum in zip(data, mask)])
            else:
                data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        for i, (self_attn, self_ff, self_scale) in enumerate(self.self_attn_layers):
            if self.cross_attn_layers[i] is not None:
                # Non-destructive popping
                datum, *data = data
                maskum, *mask = mask

                cross_attn, cross_ff, cross_scale = self.cross_attn_layers[i]
                x = cross_scale(cross_attn(x, context=datum, mask=maskum, rotary_pos_emb=rotary_pos_emb)) + x
                x = cross_scale(cross_ff(x)) + x

            x = self_scale(self_attn(x, rotary_pos_emb=rotary_pos_emb)) + x
            x = self_scale(self_ff(x)) + x

        # Data and mask list should have been fully consumed
        assert not data and not mask

        if not exists(queries):
            return x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents
        
        latents = self.decoder_scale(self.decoder_cross_attn(queries, context=x, rotary_pos_emb=rotary_pos_emb))

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_scale(self.decoder_ff(latents))

        # final linear out

        return self.to_logits(latents)

# Perceiver LM example

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        logits = self.perceiver_io(x, mask = mask, queries = x)
        return logits


if __name__ == '__main__':
    data = [
        torch.randn(4, 32, 64),
        torch.randn(4, 32, 32)
    ]
    mask = [torch.ones(4, 32, dtype=bool)] * 2

    model = PerceiverIO(depth=2, dim=[64, 32], latent_dim=128, queries_dim=16, cross_attn_interval=[0, 1], seq_dropout_prob=0.25, rotary_pos_emb=True)

    model(data, queries=torch.randn(4, 16), mask=mask)
