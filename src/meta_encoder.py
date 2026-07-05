"""
LCTE Prototype B — Meta-encoder for augmentation-invariant task embeddings.

Architecture overview
---------------------
The meta-encoder reads all demo pair token sequences for a task and
outputs a single d_model-dimensional vector that approximates the
converged per-task embedding learned by the main TinyTransformer.

A second training objective — augmentation-consistency loss — forces the
meta-encoder to produce the same embedding regardless of which augmented
view (dihedral + color permutation) of the demo pairs is presented.
This makes the embedding orientation-invariant by construction, eliminating
the need for AAIVR orientation voting.

At inference time, the meta-encoder provides a warm-start for the per-task
embedding.  Optionally, K additional gradient steps on the embedding
(transformer weights frozen) can close any remaining gap.

Training procedure
------------------
1. Run Mithil's full training pipeline.
2. Call collect_embeddings.py to save converged per-task embeddings.
3. Run train_meta_encoder.py, which:
     - Loads the converged embeddings as targets.
     - Trains MetaEncoder with:
         L = MSE(e_hat, e_target) + λ * L_aug_consistency
     where L_aug_consistency = ||meta_enc(aug_a) - meta_enc(aug_b)||²
     for two randomly chosen augmented views of the same task.

Public API
----------
  MetaEncoderConfig  — dataclass with hyperparameters
  MetaEncoder        — nn.Module, takes List[Tensor] → Tensor
  build_meta_encoder_from_main_model — initialise encoder sharing token embedding
  AugConsistencyLoss — nn.Module wrapping the consistency objective
  WarmStartResult    — datatype returned by warm_start_embedding
  warm_start_embedding — insert meta-encoder prediction + optional fine-tune
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    IO_SEPARATOR_TOKEN_ID,
    MAX_SEQ_LEN,
    VOCAB_SIZE,
    SequenceExample,
    compute_positions_3d,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetaEncoderConfig:
    """Hyperparameters for the MetaEncoder.

    Keep n_layers and d_model small — the meta-encoder is a lightweight
    module that distils per-task embeddings from demo pairs.
    """
    d_model: int = 256          # internal hidden dimension of the meta-encoder
    target_d_model: int = 768   # must match TinyTransformer.config.d_model
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = MAX_SEQ_LEN
    # Augmentation-consistency loss weight
    lambda_consistency: float = 0.1


# ---------------------------------------------------------------------------
# Building blocks (reuse from tinytransformer style)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class _FFN(nn.Module):
    def __init__(self, config: MetaEncoderConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff * 2)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.fc_in(x).chunk(2, dim=-1)
        return self.drop(self.fc_out(self.drop(x * F.silu(gate))))


class _MHA(nn.Module):
    def __init__(self, config: MetaEncoderConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop_p = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_bias = None
        if attention_mask is not None:
            # attention_mask: [B, S] bool, True = valid
            attn_bias = torch.zeros(B, 1, S, S, device=x.device, dtype=x.dtype)
            key_mask = ~attention_mask[:, None, None, :]  # True = pad
            attn_bias = attn_bias.masked_fill(key_mask, float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class _EncoderBlock(nn.Module):
    def __init__(self, config: MetaEncoderConfig) -> None:
        super().__init__()
        self.ln1 = _RMSNorm(config.d_model)
        self.attn = _MHA(config)
        self.ln2 = _RMSNorm(config.d_model)
        self.ff = _FFN(config)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Meta-Encoder
# ---------------------------------------------------------------------------

class MetaEncoder(nn.Module):
    """
    Lightweight encoder that maps a task's demo pairs → per-task embedding.

    Forward pass
    ------------
    Input : a list of N token sequences (all demo pairs for one task), each
            as a 1D LongTensor.
    Output: a [target_d_model] vector matching TinyTransformer.example_embedding.

    Pooling strategy
    ----------------
    All demo pair sequences are concatenated (with a [SEP]-like boundary),
    padded to a common length, and processed by a shallow bi-directional
    transformer.  The CLS vector (first token of each sequence is replaced
    by a learned CLS token) is mean-pooled across demo pairs and then
    projected to target_d_model.

    This is NOT autoregressively masked — the meta-encoder uses full
    bidirectional attention since we want a holistic representation, not a
    causal one.
    """

    def __init__(self, config: MetaEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Separate token embedding from the main model — keeps training clean.
        # Can be optionally initialised from the main model's token_embedding later.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.cls_token = nn.Parameter(torch.randn(config.d_model) * 0.02)
        # Simple learned positional embedding (cheaper than 3D RoPE for encoder)
        self.pos_embedding = nn.Embedding(config.max_seq_len + 1, config.d_model)

        self.blocks = nn.ModuleList([_EncoderBlock(config) for _ in range(config.n_layers)])
        self.norm = _RMSNorm(config.d_model)
        self.proj = nn.Linear(config.d_model, config.target_d_model, bias=True)
        self.drop = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _encode_sequence(
        self,
        tokens: torch.Tensor,  # [S] LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed one token sequence, prepend CLS, return (hidden, mask)."""
        S = tokens.size(0)
        device = tokens.device
        tok_emb = self.token_embedding(tokens)  # [S, d_model]
        # Prepend CLS token
        cls = self.cls_token.unsqueeze(0).to(device)  # [1, d_model]
        seq = torch.cat([cls, tok_emb], dim=0)  # [S+1, d_model]
        # Add positional embedding
        pos_ids = torch.arange(S + 1, device=device)
        seq = seq + self.pos_embedding(pos_ids)
        return seq, torch.ones(S + 1, dtype=torch.bool, device=device)

    def forward(self, token_sequences: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        token_sequences : list of [S_i] LongTensors (demo pair token sequences)

        Returns
        -------
        embedding : [target_d_model] float tensor
        """
        if not token_sequences:
            device = self.cls_token.device
            return torch.zeros(self.config.target_d_model, device=device)

        device = token_sequences[0].device
        encoded: List[torch.Tensor] = []  # list of [S_i+1, d_model]
        masks: List[torch.Tensor] = []

        for tok in token_sequences:
            seq, mask = self._encode_sequence(tok.to(device))
            encoded.append(seq)
            masks.append(mask)

        # Pad to the same length within this task's sequences
        max_len = max(e.size(0) for e in encoded)
        batch_embs = torch.zeros(len(encoded), max_len, self.config.d_model, device=device)
        batch_mask = torch.zeros(len(encoded), max_len, dtype=torch.bool, device=device)

        for i, (seq, mask) in enumerate(zip(encoded, masks)):
            L = seq.size(0)
            batch_embs[i, :L] = seq
            batch_mask[i, :L] = mask

        hidden = self.drop(batch_embs)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=batch_mask)
        hidden = self.norm(hidden)

        # Extract CLS vectors (position 0 in each sequence), pool over demo pairs
        cls_vecs = hidden[:, 0, :]  # [N_pairs, d_model]
        pooled = cls_vecs.mean(dim=0)  # [d_model]

        out = self.proj(pooled)  # [target_d_model]
        return out

    def forward_batch_tasks(
        self, task_sequences: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Process multiple tasks in sequence, return stacked embeddings.

        Parameters
        ----------
        task_sequences : list of N tasks, each a list of demo pair tensors

        Returns
        -------
        embeddings : [N, target_d_model]
        """
        return torch.stack([self.forward(seqs) for seqs in task_sequences], dim=0)


# ---------------------------------------------------------------------------
# Augmentation-consistency loss
# ---------------------------------------------------------------------------

class AugConsistencyLoss(nn.Module):
    """
    Penalise differences between embeddings of two augmented views of the same task.

    L_aug = mean over tasks of  ||e(aug_a) - e(aug_b)||^2

    where aug_a and aug_b are two independently drawn augmented views
    (different dihedral + color permutation) of the same task's demo pairs.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        emb_a: torch.Tensor,  # [N, D]
        emb_b: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        diffs = (emb_a - emb_b).pow(2).sum(dim=-1)  # [N]
        if self.reduction == "mean":
            return diffs.mean()
        return diffs.sum()


# ---------------------------------------------------------------------------
# Utility: initialise meta-encoder from main model's token embedding
# ---------------------------------------------------------------------------

def build_meta_encoder_from_main_model(
    main_model: "TinyTransformer",  # noqa: F821
    meta_config: Optional[MetaEncoderConfig] = None,
) -> MetaEncoder:
    """
    Create a MetaEncoder and copy the main model's token embedding weights.

    The token embedding provides a good initialisation because the meta-encoder
    processes the same ARC token sequences as the main model.  Sharing the
    embedding in vocabulary space (while having different model dimensions)
    requires a projection, so we copy into a separate nn.Embedding that we
    then project-down if needed.

    If meta_config.d_model == main_model.config.d_model, the token embedding
    is copied exactly.  Otherwise it is left randomly initialised.
    """
    main_cfg = main_model.config
    if meta_config is None:
        meta_config = MetaEncoderConfig(target_d_model=main_cfg.d_model)

    encoder = MetaEncoder(meta_config)

    # If dimensions match, copy the token embedding for a warm start.
    if meta_config.d_model == main_cfg.d_model:
        with torch.no_grad():
            encoder.token_embedding.weight.copy_(main_model.token_embedding.weight)
        print("MetaEncoder: copied token embedding from main model.")
    else:
        print(
            f"MetaEncoder: d_model={meta_config.d_model} ≠ main d_model={main_cfg.d_model}; "
            "token embedding is randomly initialised."
        )

    return encoder


# ---------------------------------------------------------------------------
# Warm-starting per-task embeddings at inference
# ---------------------------------------------------------------------------

@dataclass
class WarmStartResult:
    task_id: str
    predicted_embedding: torch.Tensor      # [target_d_model]
    final_embedding: torch.Tensor          # [target_d_model] after fine-tuning
    pre_finetune_loss: Optional[float]     # demo-pair output CE before fine-tune
    post_finetune_loss: Optional[float]    # demo-pair output CE after fine-tune
    n_finetune_steps: int


@torch.no_grad()
def _demo_output_ce(
    model: "TinyTransformer",  # noqa: F821
    demo_examples: List[SequenceExample],
    device: torch.device,
    dihedral_id: int = 0,
) -> float:
    """Compute output CE on demo pairs (used to measure embedding quality)."""
    from canonicalize import score_task_orientations
    _, losses = score_task_orientations(model, demo_examples, device, use_amp=True)
    return losses[dihedral_id] if losses else float("inf")


def warm_start_embedding(
    meta_encoder: MetaEncoder,
    main_model: "TinyTransformer",  # noqa: F821
    demo_examples: List[SequenceExample],
    task_id: str,
    device: torch.device,
    n_finetune_steps: int = 0,
    finetune_lr: float = 1e-2,
    dihedral_id: int = 0,
    use_amp: bool = True,
) -> WarmStartResult:
    """
    Replace the task's per-task embedding with the meta-encoder's prediction.

    Optionally, run n_finetune_steps of gradient descent on the embedding
    ONLY (transformer weights are frozen), using demo-pair output CE as the
    training signal.

    Parameters
    ----------
    meta_encoder       : trained MetaEncoder
    main_model         : TinyTransformer (weights will be modified in place!)
    demo_examples      : demo pairs with has_output=True
    task_id            : ARC task identifier
    device             : target device
    n_finetune_steps   : additional gradient steps on embedding (0 = pure warm-start)
    finetune_lr        : learning rate for embedding fine-tuning
    dihedral_id        : dihedral orientation to use for fine-tuning scoring
    use_amp            : bfloat16 autocast

    Returns
    -------
    WarmStartResult with before/after CE and the final embedding.
    """
    meta_encoder.eval()
    main_model.eval()

    valid_demos = [ex for ex in demo_examples if ex.has_output]
    if not valid_demos:
        dummy = main_model.example_embedding.weight[valid_demos[0].example_id if valid_demos else 0].detach()
        return WarmStartResult(task_id, dummy, dummy, None, None, 0)

    # Collect token sequences for the given dihedral orientation
    seqs: List[torch.Tensor] = []
    for ex in valid_demos:
        if ex.tokens_by_dihedral is not None:
            seqs.append(ex.tokens_by_dihedral[dihedral_id])
        else:
            seqs.append(ex.tokens)

    # Run meta-encoder to get predicted embedding
    with torch.no_grad():
        seqs_device = [s.to(device) for s in seqs]
        predicted = meta_encoder.forward(seqs_device).detach()  # [target_d_model]

    example_id = int(valid_demos[0].example_id)

    # Measure pre-fine-tune CE
    pre_loss: Optional[float] = None
    if n_finetune_steps > 0:
        pre_loss = _demo_output_ce(main_model, valid_demos, device, dihedral_id)

    # Insert predicted embedding
    with torch.no_grad():
        main_model.example_embedding.weight[example_id].copy_(predicted)

    if n_finetune_steps == 0:
        return WarmStartResult(task_id, predicted, predicted.clone(), pre_loss, None, 0)

    # Fine-tune the embedding (only) using demo-pair output CE
    # Freeze everything except the target embedding slot.
    from canonicalize import score_task_orientations

    emb_param = nn.Parameter(predicted.clone().to(device))
    optimizer = torch.optim.AdamW([emb_param], lr=finetune_lr, weight_decay=0.0)

    for step in range(n_finetune_steps):
        # Temporarily install current emb_param in the model
        with torch.no_grad():
            main_model.example_embedding.weight[example_id].copy_(emb_param.data)

        # Forward pass (need gradients through emb_param)
        main_model.train()
        # Accumulate CE from all demo pairs at dihedral_id
        total_loss = None
        tokens_list, positions_list, sep_list, seq_lens = [], [], [], []
        for ex in valid_demos:
            if ex.tokens_by_dihedral is not None:
                tok = ex.tokens_by_dihedral[dihedral_id].to(device)
                pos = ex.cached_positions_by_dihedral[dihedral_id].to(device)
                sep = ex.sep_index_by_dihedral[dihedral_id]
            else:
                tok = ex.tokens.to(device)
                pos = getattr(ex, "cached_positions", None)
                if pos is None:
                    fb = tok.unsqueeze(0)
                    pos = compute_positions_3d(fb, torch.ones_like(fb, dtype=torch.bool)).squeeze(0)
                pos = pos.to(device)
                sep = ex.sep_index
            tokens_list.append(tok)
            positions_list.append(pos)
            sep_list.append(sep)
            seq_lens.append(tok.size(0))

        packed_ids = torch.cat(tokens_list)
        packed_pos = torch.cat(positions_list)
        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
        ex_ids = torch.tensor([example_id] * len(valid_demos), dtype=torch.long, device=device)
        dih_ids = torch.tensor([dihedral_id] * len(valid_demos), dtype=torch.long, device=device)
        sep_t = torch.tensor(sep_list, dtype=torch.long, device=device)

        # We need gradients through the example embedding only
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            outputs = main_model(
                packed_ids, ex_ids, dih_ids,
                sep_indices=sep_t,
                compute_input_loss=False,
                positions_3d=packed_pos,
                cu_seqlens=cu_seqlens,
                max_seqlen=max(seq_lens),
            )
        step_loss = outputs.get("output_loss")
        if step_loss is None:
            break

        optimizer.zero_grad()
        step_loss.backward()
        # Manually copy the gradient for emb_param from the embedding table
        grad = main_model.example_embedding.weight.grad
        if grad is not None:
            emb_param.grad = grad[example_id].clone().to(dtype=emb_param.dtype)

        optimizer.step()

        # Zero the table's gradient
        if main_model.example_embedding.weight.grad is not None:
            main_model.example_embedding.weight.grad.zero_()

    main_model.eval()

    # Install final embedding
    final_emb = emb_param.detach()
    with torch.no_grad():
        main_model.example_embedding.weight[example_id].copy_(final_emb)

    post_loss = _demo_output_ce(main_model, valid_demos, device, dihedral_id)

    return WarmStartResult(
        task_id=task_id,
        predicted_embedding=predicted,
        final_embedding=final_emb,
        pre_finetune_loss=pre_loss,
        post_finetune_loss=post_loss,
        n_finetune_steps=n_finetune_steps,
    )
