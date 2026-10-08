"""Batched KV cache for the processor's full-attention small model."""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


def _set_layer_kv(layer, k, v):
    # Release the old chunk buffer as soon as its logical KV tensors are replaced.
    # The next update rebuilds the buffer from the new tensors.
    layer._buf = None
    layer.keys, layer.values = k, v


def _zpad(k, v, n):
    """Left-pad a layer's keys/values with n zero columns (n <= 0: unchanged)."""
    if n <= 0: return k, v
    z = torch.zeros(k.shape[0], k.shape[1], n, k.shape[3], dtype=k.dtype, device=k.device)
    return torch.cat([z, k], 2), torch.cat([torch.zeros_like(z), v], 2)


class ChunkedLayer(DynamicLayer):
    """Grow full-attention KV storage in chunks instead of copying it every token.

    Merge, eviction and trimming replace the logical view; the next update rebuilds
    the backing buffer from that view.
    """
    CHUNK = 1024

    def __init__(self, *a, **k):
        super().__init__(*a, **k); self._buf = None
    @classmethod
    def from_layer(cls, layer):
        c = cls()
        for attr in ("keys", "values", "dtype", "device", "is_initialized"):
            if hasattr(layer, attr): setattr(c, attr, getattr(layer, attr))
        return c
    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized: self.lazy_initialization(key_states, value_states)
        k, v = self.keys, self.values
        L = 0 if k.numel() == 0 else k.shape[2]
        B, H, T, D = key_states.shape
        b = self._buf
        ok = (b is not None and L == b[2] and L + T <= b[0].shape[2] and b[0].shape[0] == B and b[0].shape[1] == H
              and b[0].shape[3] == D and k.data_ptr() == b[0].data_ptr() and k.device == b[0].device and k.dtype == b[0].dtype)
        if not ok:
            cap = L + T + self.CHUNK
            kb = key_states.new_empty((B, H, cap, D)); vb = value_states.new_empty((B, H, cap, D))
            if L: kb[:, :, :L] = k; vb[:, :, :L] = v
            b = (kb, vb, L)
        kb, vb, _ = b
        kb[:, :, L:L + T] = key_states; vb[:, :, L:L + T] = value_states
        self._buf = (kb, vb, L + T)
        self.keys, self.values = kb[:, :, :L + T], vb[:, :, :L + T]
        return self.keys, self.values


def _chunkify(cache):
    """Swap the full-attention layers of a freshly prefilled DynamicCache for ChunkedLayer."""
    cache.layers = [ChunkedLayer.from_layer(layer) for layer in cache.layers]
    return cache



class ModelStream:
    """One model's batch of rows with a shared, continuously maintained KV cache."""

    def __init__(self, model, pad_id, device):
        self.m, self.pad, self.dev = model, pad_id, device
        self.rows = []            # row ids in batch order
        self.logits = None        # R x V float32, last-position logits
        self.past = self.att = self.nextpos = None
        self._next_id = 0

    # ---- capacity ----
    def width(self): return 0 if self.att is None else int(self.att.shape[1])

    # ---- admission ----
    @torch.no_grad()
    def admit(self, prompts):
        """Prefill new rows (left-padded among themselves) and merge them into the batch."""
        lens = [int(p.shape[0]) for p in prompts]
        L = max(lens)
        inp = torch.full((len(prompts), L), self.pad, dtype=torch.long, device=self.dev)
        att = torch.zeros((len(prompts), L), dtype=torch.long, device=self.dev)
        for i, p in enumerate(prompts):
            inp[i, L - lens[i]:] = p.to(self.dev); att[i, L - lens[i]:] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        o = self.m(input_ids=inp, attention_mask=att, position_ids=pos, past_key_values=DynamicCache(config=self.m.config), use_cache=True, logits_to_keep=1)
        nl, npast = o.logits[:, -1, :].float().to(self.dev), o.past_key_values
        npos = torch.tensor(lens, device=self.dev, dtype=torch.long).view(-1, 1)
        ids = list(range(self._next_id, self._next_id + len(prompts))); self._next_id += len(prompts)
        if self.past is None:
            self.logits, self.past, self.att, self.nextpos = nl, _chunkify(npast), att, npos
        else:
            self._merge(nl, npast, att, npos)
        self.rows += ids
        return ids

    def _merge(self, nl, npast, natt, npos):
        La, Lb = self.att.shape[1], natt.shape[1]; L = max(La, Lb)
        if La < L: self.att = self._pad_att(self.att, L - La)
        if Lb < L: natt = self._pad_att(natt, L - Lb)
        for la, lb in zip(self.past.layers, npast.layers):
            ka, va = _zpad(la.keys, la.values, L - la.keys.shape[2])
            kb, vb = _zpad(lb.keys, lb.values, L - lb.keys.shape[2])
            _set_layer_kv(la, torch.cat([ka, kb], 0), torch.cat([va, vb], 0))
        self.att = torch.cat([self.att, natt], 0)
        self.nextpos = torch.cat([self.nextpos, npos], 0)
        self.logits = torch.cat([self.logits, nl], 0)

    def _pad_att(self, att, n):
        return torch.cat([torch.zeros(att.shape[0], n, dtype=att.dtype, device=self.dev), att], 1)


    # ---- eviction ----
    def evict(self, row_ids):
        if not row_ids: return
        drop = set(row_ids)
        keep = [i for i, r in enumerate(self.rows) if r not in drop]
        self.rows = [r for r in self.rows if r not in drop]
        if not keep:
            self.logits = self.past = self.att = self.nextpos = None
            return
        idx = torch.tensor(keep, device=self.dev, dtype=torch.long)
        for layer in self.past.layers:
            _set_layer_kv(layer, layer.keys[idx], layer.values[idx])
        self.att, self.nextpos, self.logits = self.att[idx], self.nextpos[idx], self.logits[idx]
        # drop left-padding columns that no surviving row uses
        live = self.att.sum(0).nonzero()
        first = int(live[0]) if live.numel() else 0
        if first > 0:
            self.att = self.att[:, first:]
            for layer in self.past.layers:
                k, v = layer.keys, layer.values
                _set_layer_kv(layer, k[:, :, first:, :].contiguous(), v[:, :, first:, :].contiguous())

    # ---- one decode step for every row ----
    @torch.no_grad()
    def step(self, tokens_by_row):
        """Advance each row by its supplied token or token list; an empty list stalls it.
        
        Rows share a padded width, so shorter chunks are left-padded without advancing their
        real positions. A stalled row keeps its previous logits.
        """
        seqs, kr = [], []
        for r in self.rows:
            v = tokens_by_row[r]
            q = list(v) if isinstance(v, (list, tuple)) else [v]
            seqs.append(q); kr.append(len(q))
        k = max(kr) if kr else 0
        if k < 1:
            return                                   # every row stalls: nothing to feed this step
        seqs = [[self.pad] * (k - len(q)) + q for q in seqs]   # pads first, real tokens end at the last column
        t = torch.tensor(seqs, device=self.dev, dtype=torch.long)                         # R x k
        krt = torch.tensor(kr, device=self.dev, dtype=torch.long).view(-1, 1)             # R x 1
        ar = torch.arange(k, device=self.dev, dtype=torch.long).view(1, k)
        m = (ar >= (k - krt)).to(self.att.dtype)                                          # 1 on the real tokens
        self.att = torch.cat([self.att, m], 1)
        pos = (self.nextpos - (k - krt) + ar).clamp(min=0)                                # pads: any position, masked
        o = self.m(input_ids=t, attention_mask=self.att, position_ids=pos,
                   past_key_values=self.past, use_cache=True, logits_to_keep=1)
        nl = o.logits[:, -1, :].float().to(self.dev)
        if min(kr) == 0:                             # a stalled row's last column is a pad: keep its logits
            nl = torch.where((krt == 0), self.logits, nl)
        self.logits, self.past = nl, o.past_key_values
        self.nextpos = self.nextpos + krt

    def logits_of(self, row_id):
        return self.logits[self.rows.index(row_id)]
