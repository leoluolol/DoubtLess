"""Base and brief/verbose small-model passes within one vLLM request.

The base sees P T G: prompt, assistant header, generated text. Each small-model
pass sees P S T G, with its instruction S inserted before the header. P, T and G
use the request's paged KV blocks; S uses a separate instruction buffer.
FlashAttention and instruction attention are merged by their log-sum-exp weights.

After prefilling P, the metadata builder prefills S. compute_logits then forces
the header T before returning contrast logits to the sampler. Small-model
positions after P are shifted by |S|. sched.py splits preemption replay at |P|
to restore S before replaying generated tokens.

EOS and the closing tag retain their base logits. After the closing tag is fed,
sampling uses the base logits. The header contains an end-of-turn token, so
engine.py remaps that token to a separate stop ID and strips the forced header.

The small decoder supports full-attention Qwen3 and Llama/Qwen2. Its brief and
verbose passes share weights and run as one doubled batch, captured by full
decode CUDA graphs. engine.py configures the graphs and shared attention backend.
"""
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata,
                                                   FlashAttentionMetadataBuilder)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from doubtless.contrast import NEG, combine
from doubtless.vllm.instruction_attn import instruction_attention_triton
from doubtless.vllm.state import S_MAX, ThinkerState, plan_step
from doubtless.vllm import sched as _sched
_sched.install()

# ============================================================================================ attention backend
@dataclass
class ThinkerAttentionMetadata(FlashAttentionMetadata):
    """FlashAttention's metadata plus, per scheduled token, what the thinker attention needs.  All tensors are views
    of persistent buffers owned by the builder, so a FULL CUDA graph captured with them replays correctly."""
    thinker_instr_prefill: bool = False                    # instruction prefill (queries = S tokens; keys = paged P + causal S)
    thinker_kv_row: torch.Tensor | None = None   # (N,) instruction-buffer row of the token's request
    thinker_instr_len: torch.Tensor | None = None   # (2, N) instruction keys visible to the token, per role
    thinker_instr_pos: torch.Tensor | None = None   # (N,) S-mode: the token's index within S
    thinker_pos: torch.Tensor | None = None        # (2, N) thinker positions per role (base position + shift)


class ThinkerMetadataBuilder(FlashAttentionMetadataBuilder):
    """Built once per step for the thinker attention group (before the forward, on the host; FULL-graph replay
    steps included).  Adds the per-token fields above and performs the instruction prefill for
    requests whose prompt has just been computed."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.thinker_layer_names = list(layer_names)
        cap = int(vllm_config.scheduler_config.max_num_batched_tokens) + 8
        self._kv_row = torch.zeros(cap, dtype=torch.long, device=device)
        self._instr_len = torch.zeros(2, cap, dtype=torch.int32, device=device)
        self._pos = torch.zeros(2, cap, dtype=torch.int64, device=device)
        self._req_of = torch.zeros(cap, dtype=torch.int32, device=device)      # int32: token_to_req_indices asserts it
        self._zero_pos = torch.zeros(cap, dtype=torch.int32, device=device)

    def _to_thinker(self, md, n):
        if isinstance(md, ThinkerAttentionMetadata):
            return md
        kw = {f.name: getattr(md, f.name) for f in dataclasses.fields(md)}
        return ThinkerAttentionMetadata(**kw, thinker_kv_row=self._kv_row[:n], thinker_instr_len=self._instr_len[:, :n],
                                   thinker_instr_pos=self._zero_pos[:n], thinker_pos=self._pos[:, :n])

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cm = common_attn_metadata
        st = ThinkerState.get()
        n = int(cm.num_actual_tokens)
        qsl_cpu = cm.query_start_loc_cpu.tolist()
        query_lens = [qsl_cpu[i + 1] - qsl_cpu[i] for i in range(len(qsl_cpu) - 1)]
        seq_lens_cpu = cm.seq_lens_cpu_upper_bound.tolist()          # the supported host copy
        num_computed = [s - q for s, q in zip(seq_lens_cpu, query_lens)]
        n_req = len(query_lens)
        # Prefill instructions after the prompt; sched.py splits chunks at this boundary.
        due, crossing = plan_step(query_lens, num_computed, st.prompt_len_host, st.pending)
        if crossing:
            raise RuntimeError("doubtless: a recompute chunk crosses the prompt end for rows %s; "
                               "the scheduler hook is not active in the engine-core process" % crossing)
        if due:
            per_call = max(1, int(self.vllm_config.scheduler_config.max_num_batched_tokens) // (2 * S_MAX))
            for c in range(0, len(due), per_call):
                self.instruction_prefill(due[c:c + per_call], cm.block_table_tensor)
        # 2. per-token fields: request row, thinker positions, instruction visibility; and the rows' current length
        req_of = cm.token_to_req_indices(self._req_of).long()
        rows = req_of
        pos = cm.positions[:n]
        gen = pos >= st.prompt_len[rows].to(pos.dtype)               # tokens after the prompt see S and shift
        for role in (0, 1):
            sl = st.instr_len[rows, role]
            self._pos[role, :n] = pos.to(torch.int64) + torch.where(gen, sl.to(torch.int64), torch.zeros_like(sl, dtype=torch.int64))
            self._instr_len[role, :n] = torch.where(gen, sl, torch.zeros_like(sl))
        self._kv_row[:n] = st.kv_row[rows]
        st.cur_len[:n_req] = cm.seq_lens[:n_req].to(torch.int32)       # length after this step, per batch row
        return self._to_thinker(md, n)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        md = self.build(0, common_attn_metadata)                     # already a ThinkerAttentionMetadata
        n = int(common_attn_metadata.num_actual_tokens)
        self._instr_len[:, :n].zero_()
        self._kv_row[:n].zero_()
        self._pos[:, :n] = common_attn_metadata.positions[:n].to(torch.int64)
        return md

    # ---- instruction prefill ------------------------------------------------------------------------------
    def instruction_prefill(self, rows, block_table):
        """Run the thinker decoder over the instruction tokens of `rows` against their paged prompt and write the
        K/V into the instruction buffer.  Both roles are padded to S_MAX so the model's two halves have equal size.
        Runs eagerly under its own forward context (cudagraph mode NONE, slot mapping -1: no paged write)."""
        st = ThinkerState.get()
        model = st.model
        k = len(rows)
        dev = self.device
        rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
        m = k * S_MAX
        ids = torch.cat([st.instr[rows_t, 0].reshape(-1), st.instr[rows_t, 1].reshape(-1)])          # (2m,)
        j = torch.arange(S_MAX, device=dev)
        pos_role = (st.prompt_len[rows_t].to(torch.int64).unsqueeze(1) + j.unsqueeze(0)).reshape(-1)   # (m,)
        positions = torch.cat([pos_role, pos_role])
        qsl = torch.arange(0, m + 1, S_MAX, dtype=torch.int32, device=dev)
        seq_lens = st.prompt_len[rows_t].to(torch.int32)
        max_seq = int(max(st.prompt_len_host[r] for r in rows))
        bt = block_table[rows_t]
        slot_map = torch.full((m,), -1, dtype=torch.int64, device=dev)
        side_row = st.kv_row[rows_t].repeat_interleave(S_MAX)
        side_pos = j.repeat(k).to(torch.int32)
        side_len = torch.stack([torch.minimum(side_pos + 1, st.instr_len[rows_t, role].repeat_interleave(S_MAX))
                                for role in (0, 1)])                                   # causal within S, per role
        md = ThinkerAttentionMetadata(num_actual_tokens=m, max_query_len=S_MAX, query_start_loc=qsl, max_seq_len=max_seq,
                                 seq_lens=seq_lens, block_table=bt, slot_mapping=slot_map,
                                 use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
                                 prefix_kv_lens=None, suffix_kv_lens=None, causal=False,
                                 thinker_instr_prefill=True, thinker_kv_row=side_row, thinker_instr_len=side_len,
                                 thinker_instr_pos=side_pos, thinker_pos=torch.stack([pos_role, pos_role]))
        attn = {name: md for name in self.thinker_layer_names}
        slots = {name: slot_map for name in self.thinker_layer_names}
        with set_forward_context(attn, self.vllm_config, num_tokens=2 * m, slot_mapping=slots):
            model.thinker(input_ids=ids, positions=positions)
        for r in rows:
            st.pending.discard(r)


class ThinkerAttentionImpl(FlashAttentionImpl):
    """Thinker attention: paged FlashAttention over the request's blocks (with LSE) merged with dense attention over
    the request's instruction K/V in the instruction buffer.  In S-mode the queries are the instruction tokens themselves:
    their K/V go to the instruction buffer, the paged keys are the prompt only and the instruction part is causal within S.
    query/key/value/output arrive as (tokens, heads, head_dim)."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:                      # profiling / dummy runs
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("doubtless: fused output quantisation is not supported")
        md = attn_metadata
        n = md.num_actual_tokens
        role, layer_idx = layer.thinker_role, layer.thinker_layer
        st = ThinkerState.get()
        q = query[:n]
        side = st.instr_kv[(role, layer_idx)]
        if md.thinker_instr_prefill:
            # write this token's K/V into its request's buffer row.  All S_MAX positions are written (the padded
            # ones hold keys nobody reads: visibility is capped by thinker_instr_len), so no data-dependent indexing
            # and no lookup by buffer row -- instr_len is indexed by batch slot, not by buffer row
            p_sel = md.thinker_instr_pos.long()
            side[md.thinker_kv_row, 0, p_sel] = key[:n]
            side[md.thinker_kv_row, 1, p_sel] = value[:n]
        # Paged attention returns LSE for merging with instruction attention.
        # vLLM selects the FlashAttention version; FA3/FA4 choose their own split count.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        out1, lse1 = flash_attn_varlen_func(
            q=q, k=key_cache, v=value_cache, cu_seqlens_q=md.query_start_loc, max_seqlen_q=md.max_query_len,
            seqused_k=md.seq_lens, max_seqlen_k=md.max_seq_len, softmax_scale=self.scale, causal=bool(md.causal),
            block_table=md.block_table, return_softmax_lse=True, fa_version=self.vllm_flash_attn_version)
        # The kernel gathers instruction rows directly, including during large prefills.
        lens = md.thinker_instr_len[role]
        out2, lse2 = instruction_attention_triton(q, side, md.thinker_kv_row, lens, self.scale)
        merge_attn_states(output[:n], out1, lse1.to(torch.float32), out2.to(out1.dtype), lse2)
        return output


class ThinkerAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return ThinkerMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return ThinkerAttentionImpl


# ============================================================================================ thinker decoder
import vllm._custom_ops as _ops


class FusedRMSNorm(nn.Module):
    """Use fused RMSNorm for the eager small decoder, including residual addition."""

    def __init__(self, hidden, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = float(eps)

    def forward(self, x, residual=None):
        if residual is None:
            out = torch.empty_like(x)
            _ops.rms_norm(out, x, self.weight, self.eps)
            return out
        _ops.fused_add_rms_norm(x, residual, self.weight, self.eps)     # x <- norm(x + residual), residual <- x + residual
        return x, residual


class ThinkerAttention(nn.Module):
    def __init__(self, hf, vllm_config, prefix, layer_idx, max_position):
        super().__init__()
        tp = get_tensor_model_parallel_world_size()
        self.total_heads, self.total_kv = hf.num_attention_heads, hf.num_key_value_heads
        self.heads = self.total_heads // tp
        self.kv_heads = max(1, self.total_kv // tp)
        self.head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        self.q_size, self.kv_size = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_proj = QKVParallelLinear(hf.hidden_size, self.head_dim, self.total_heads, self.total_kv,
                                          bias=getattr(hf, "attention_bias", hf.model_type == "qwen2"),
                                          prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(self.total_heads * self.head_dim, hf.hidden_size, bias=False, prefix=f"{prefix}.o_proj")
        self.qk_norm = thinker_family(hf) == "qwen3"
        if self.qk_norm:
            self.q_norm = FusedRMSNorm(self.head_dim, hf.rms_norm_eps)
            self.k_norm = FusedRMSNorm(self.head_dim, hf.rms_norm_eps)
        self.eps = float(hf.rms_norm_eps)
        self.rotary_emb = get_rope(self.head_dim, max_position=max_position, rope_parameters=hf.rope_parameters)
        cache_config = vllm_config.cache_config
        self.attn_b = Attention(self.heads, self.head_dim, self.scaling, num_kv_heads=self.kv_heads,
                                cache_config=cache_config, prefix=f"{prefix}.attn_b", attn_backend=ThinkerAttentionBackend)
        self.attn_v = Attention(self.heads, self.head_dim, self.scaling, num_kv_heads=self.kv_heads,
                                cache_config=cache_config, prefix=f"{prefix}.attn_v", attn_backend=ThinkerAttentionBackend)
        for role, a in ((0, self.attn_b), (1, self.attn_v)):
            a.thinker_role, a.thinker_layer = role, layer_idx

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        if not self.qk_norm:                                         # Llama / Qwen2-style thinker: RoPE only
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
        else:
            # q_norm, k_norm and RoPE in one kernel, in place on the packed qkv (vLLM's fused_qk_norm_rope)
            _ops.fused_qk_norm_rope(qkv, self.heads, self.kv_heads, self.kv_heads, self.head_dim, self.eps,
                                    self.q_norm.weight, self.k_norm.weight, self.rotary_emb.cos_sin_cache, True, positions)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        m = q.shape[0] // 2                                          # [brief rows ; verbose rows]
        o = torch.cat([self.attn_b(q[:m], k[:m], v[:m]), self.attn_v(q[m:], k[m:], v[m:])], dim=0)
        out, _ = self.o_proj(o)
        return out


class ThinkerMLP(nn.Module):
    def __init__(self, hf, prefix):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hf.hidden_size, [hf.intermediate_size] * 2, bias=False,
                                                       prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(hf.intermediate_size, hf.hidden_size, bias=False, prefix=f"{prefix}.down_proj")
        self.act = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act(x))
        return x


class ThinkerLayer(nn.Module):
    def __init__(self, hf, vllm_config, prefix, layer_idx, max_position):
        super().__init__()
        self.self_attn = ThinkerAttention(hf, vllm_config, f"{prefix}.self_attn", layer_idx, max_position)
        self.mlp = ThinkerMLP(hf, f"{prefix}.mlp")
        self.input_layernorm = FusedRMSNorm(hf.hidden_size, hf.rms_norm_eps)
        self.post_attention_layernorm = FusedRMSNorm(hf.hidden_size, hf.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class ThinkerDecoder(nn.Module):
    """Small decoder over [brief; verbose], captured by the full decode CUDA graphs."""

    def __init__(self, *, vllm_config: VllmConfig, hf, prefix: str = "thinker"):
        super().__init__()
        self.hf = hf
        max_position = max(int(hf.max_position_embeddings), int(vllm_config.model_config.max_model_len) + S_MAX + 8)
        self.embed_tokens = VocabParallelEmbedding(hf.vocab_size, hf.hidden_size, prefix=f"{prefix}.embed_tokens")
        self.layers = nn.ModuleList([ThinkerLayer(hf, vllm_config, f"{prefix}.layers.{i}", i, max_position)
                                     for i in range(hf.num_hidden_layers)])
        self.norm = FusedRMSNorm(hf.hidden_size, hf.rms_norm_eps)

    def forward(self, input_ids, positions):
        h = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            h, residual = layer(positions, h, residual)
        h, _ = self.norm(h, residual)
        return h


# ============================================================================================ the model
KNOWN_BASES = {   # resolved without the registry when possible; anything else goes through ModelRegistry
    "Qwen3ForCausalLM": "vllm.model_executor.models.qwen3:Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "vllm.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM",
    "Qwen2ForCausalLM": "vllm.model_executor.models.qwen2:Qwen2ForCausalLM",
    "LlamaForCausalLM": "vllm.model_executor.models.llama:LlamaForCausalLM",
}


def base_class(arch, vllm_config=None):
    """The vLLM model class of the base architecture (the checkpoint's own `architectures[0]`, which the engine
    passes as additional_config["base_arch"] before selecting the contrast model)."""
    import importlib
    if arch in KNOWN_BASES:
        mod, cls = KNOWN_BASES[arch].split(":")
        return getattr(importlib.import_module(mod), cls)
    from vllm import ModelRegistry
    try:
        cls, _ = ModelRegistry.resolve_model_cls([arch], vllm_config.model_config)
    except TypeError:
        cls, _ = ModelRegistry.resolve_model_cls([arch])
    return cls


def thinker_family(hf):
    """Qwen3 uses per-head q/k norms; full-attention Llama/Qwen2 omit them."""
    mt = str(getattr(hf, "model_type", "")).lower()
    if "sliding_attention" in (getattr(hf, "layer_types", None) or []) or getattr(hf, "use_sliding_window", False):
        raise NotImplementedError("doubtless.vllm requires a full-attention small model")
    if mt == "qwen3":
        return "qwen3"
    if mt in ("llama", "qwen2"):
        return "llama"
    raise NotImplementedError("doubtless.vllm: no thinker decoder for model_type %r (supported: qwen3, llama, qwen2); "
                              "use doubtless.processor for this family" % mt)


class ContrastForCausalLM(nn.Module):
    STACKED = [("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v"),
               ("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = dict(vllm_config.additional_config or {})
        self.cfg = cfg
        self.base = base_class(cfg.get("base_arch", "Qwen3ForCausalLM"), vllm_config)(vllm_config=vllm_config, prefix=prefix)
        small = cfg["small"]
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(small)
        set_default_rope_theta(hf, default_theta=float(getattr(hf, "rope_theta", 1000000)))   # transformers >= 5.10 keeps it in rope_parameters
        self.thinker = ThinkerDecoder(vllm_config=vllm_config, hf=hf, prefix="thinker")
        self.thinker_lm_head = ParallelLMHead(hf.vocab_size, hf.hidden_size, prefix="thinker_lm_head")
        self.tied = bool(getattr(hf, "tie_word_embeddings", False))
        if self.tied:
            self.thinker_lm_head = self.thinker_lm_head.tie_weights(self.thinker.embed_tokens)
        self.thinker_logits = LogitsProcessor(hf.vocab_size)
        self.secondary_weights = [DefaultModelLoader.Source(small, revision=None, prefix="thinker.")]
        self.h_base = int(vllm_config.model_config.hf_config.hidden_size)
        self.h_small = int(hf.hidden_size)
        self.vocab = min(int(vllm_config.model_config.hf_config.vocab_size), int(hf.vocab_size))
        self.p_keep = float(cfg.get("p_keep", 0.95))
        self.eos_ids = [int(t) for t in cfg.get("eos_ids", [])]
        self.close_ids = [int(t) for t in cfg.get("close_ids", [])]
        self.tail_ids = [int(t) for t in cfg.get("tail_ids", [])]
        self.end_turn_id = int(cfg["end_turn_id"]) if "end_turn_id" in cfg else None     # <|im_end|>, forced as T[0]
        self.stop_id = int(cfg["stop_id"]) if "stop_id" in cfg else None                 # <|endoftext|>, the one stop id
        self.tail = None
        self._thinker_stream = None
        st = ThinkerState.get()                       # created by ThinkerStateProcessor (the runner builds it before the model)
        st.model = self
        st.cur_len = torch.zeros(st.max_slots, dtype=torch.int32, device=st.device)
        tp = get_tensor_model_parallel_world_size()
        head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
        # allocated here, before vLLM profiles memory, so the KV budget accounts for it
        st.alloc_instruction_kv(hf.num_hidden_layers, max(1, hf.num_key_value_heads // tp), head_dim, vllm_config.model_config.dtype)
        self.make_empty_intermediate_tensors = self.base.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.base.embed_input_ids(input_ids)

    # ---- forward ------------------------------------------------------------------------------------------
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        n = input_ids.shape[0]
        st = ThinkerState.get()
        ctx = get_forward_context()
        md = None
        if isinstance(ctx.attn_metadata, dict):
            md = ctx.attn_metadata.get(self.thinker.layers[0].self_attn.attn_b.layer_name)
        if md is not None and getattr(md, "thinker_pos", None) is not None:
            assert md.num_actual_tokens == n, "doubtless: metadata for %d tokens, inputs %d" % (md.num_actual_tokens, n)
            pos2 = torch.cat([md.thinker_pos[0, :n], md.thinker_pos[1, :n]])
            # feed the last token of every scheduled request into the closing ring (generated tokens only;
            # padding rows under FULL graphs have zero-length queries and are masked out, no host sync)
            qsl = md.query_start_loc
            last = (qsl[1:] - 1).clamp(min=0)
            valid = qsl[1:] > qsl[:-1]
            rows = torch.arange(last.shape[0], device=input_ids.device)
            st.feed(rows, input_ids[last], positions[last], valid)
        else:                                      # profiling / dummy runs: no metadata
            pos2 = torch.cat([positions, positions])
        ids2 = torch.cat([input_ids, input_ids])
        if input_ids.is_cuda:
            # The two models only share the input token: run the thinker on a second stream while the base streams
            # its weights (the base is bandwidth-bound; the thinker is a few hundred small kernels).  Fork/join with
            # stream waits keeps the order, and a FULL CUDA graph captures the two branches in parallel.
            if self._thinker_stream is None:
                self._thinker_stream = torch.cuda.Stream(device=input_ids.device)
            cur = torch.cuda.current_stream(input_ids.device)
            self._thinker_stream.wait_stream(cur)
            with torch.cuda.stream(self._thinker_stream):
                h2 = self.thinker(input_ids=ids2, positions=pos2)
            h_base = self.base(input_ids, positions, intermediate_tensors, inputs_embeds)
            cur.wait_stream(self._thinker_stream)
        else:
            h_base = self.base(input_ids, positions, intermediate_tensors, inputs_embeds)
            h2 = self.thinker(input_ids=ids2, positions=pos2)
        return torch.cat([h_base, h2[:n], h2[n:]], dim=-1)

    # ---- logits -------------------------------------------------------------------------------------------
    def compute_logits(self, hidden_states):
        hb, hbr, hv = torch.split(hidden_states, [self.h_base, self.h_small, self.h_small], dim=-1)
        zb = self.base.logits_processor(self.base.lm_head, hb.contiguous())
        if zb is None:
            return None
        zz = self.thinker_logits(self.thinker_lm_head, torch.cat([hbr, hv], dim=0).contiguous())   # both roles at once
        zbr, zv = zz[: hbr.shape[0]], zz[hbr.shape[0]:]
        V = self.vocab
        zb, zbr, zv = zb[:, :V].float(), zbr[:, :V].float(), zv[:, :V].float()
        st = ThinkerState.get()
        m = zb.shape[0]
        rows = torch.arange(m, device=zb.device)
        alpha = st.alpha[rows] * st.open_rows(rows).to(zb.dtype)          # 0 outside the thinking block
        z = combine(zb, zbr, zv, alpha, self.p_keep, self.eos_ids, self.close_ids)
        if self.tail_ids:
            if self.tail is None:
                self.tail = torch.tensor(self.tail_ids, dtype=torch.long, device=zb.device)
            g = (st.cur_len[rows] - st.prompt_len[rows]).long()
            live = st.active[rows]
            # the template tail: forced one token per step right after the prompt
            force = live & (g >= 0) & (g < len(self.tail_ids))
            tok = self.tail[g.clamp(0, len(self.tail_ids) - 1)]
            forced = torch.full_like(zb, NEG)
            forced[rows, tok] = 0.0
            z = torch.where(force.unsqueeze(1), forced, z)
            # past the tail: the request runs with ignore_eos (T[0] is <|im_end|>), so the end-of-turn logit
            # moves onto the single stop id; the driver maps it back when it records the run
            if self.end_turn_id is not None and self.stop_id is not None:
                past = live & (g >= len(self.tail_ids))
                end = z[:, self.end_turn_id]
                z[:, self.stop_id] = torch.where(past, end, z[:, self.stop_id])
                z[:, self.end_turn_id] = torch.where(past, torch.full_like(end, NEG), end)
        return z

    # ---- weights ------------------------------------------------------------------------------------------
    def load_weights(self, weights):
        thinker_w = []

        def base_stream():
            for name, w in weights:
                if name.startswith("thinker."):
                    thinker_w.append((name[len("thinker."):], w))
                else:
                    yield name, w

        loaded = {"base." + n for n in self.base.load_weights(base_stream())}
        loaded |= self._load_thinker(thinker_w)
        return loaded

    def _load_thinker(self, weights):
        params = dict(self.thinker.named_parameters())
        head = dict(self.thinker_lm_head.named_parameters())
        loaded = set()
        for name, w in weights:
            if name.startswith("model."):
                name = name[len("model."):]
            if name == "lm_head.weight":
                if not self.tied and "weight" in head:
                    p = head["weight"]
                    getattr(p, "weight_loader", default_weight_loader)(p, w)
                    loaded.add("thinker_lm_head.weight")
                continue
            hit = False
            for pname, sname, sid in self.STACKED:
                if sname in name:
                    tgt = name.replace(sname, pname)
                    if tgt in params:
                        params[tgt].weight_loader(params[tgt], w, sid)
                        loaded.add("thinker." + tgt); hit = True
                    break
            if hit:
                continue
            if name in params:
                p = params[name]
                getattr(p, "weight_loader", default_weight_loader)(p, w)
                loaded.add("thinker." + name)
        return loaded
