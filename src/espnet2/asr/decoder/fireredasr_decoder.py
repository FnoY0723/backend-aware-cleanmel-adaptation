
"""FiredASR Transformer Decoder definition."""
import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.legacy.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet2.legacy.nets.scorer_interface import BatchScorerInterface


class FiredASRTransformerDecoder(AbsDecoder, BatchScorerInterface):
    """FiredASR Transformer Decoder for ESPnet training.

    Layer names match the original TransformerDecoder so pretrained
    weights can be loaded directly via load_state_dict.

    Args:
        vocab_size: output vocabulary size
        encoder_output_size: encoder output dimension (d_model)
        attention_heads: number of attention heads
        num_blocks: number of decoder layers
        dropout_rate: dropout rate
        pe_maxlen: max length for positional encoding
        use_output_layer: whether to use output projection (tgt_word_prj)
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 20,
        num_blocks: int = 16,
        dropout_rate: float = 0.0,
        pe_maxlen: int = 5000,
        use_output_layer: bool = True,
        softmax_smoothing: float = 1.0,
        sos_id: int = -1,
        eos_id: int = -1,
        pad_id: int = -1,
    ):
        super().__init__()
        d_model = encoder_output_size
        self.INF = 1e10
        self.n_layers = num_blocks
        self.scale = d_model ** 0.5
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.tgt_word_emb = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, attention_heads, dropout_rate) for _ in range(num_blocks)]
        )
        self.layer_norm_out = nn.LayerNorm(d_model)

        if use_output_layer:
            self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
        else:
            self.tgt_word_prj = None

        self.softmax_smoothing = softmax_smoothing

        logging.info(
            f"FiredASRTransformerDecoder: d_model={d_model}, "
            f"n_layers={num_blocks}, n_head={attention_heads}"
        )


    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forcing forward pass (training).

        Args:
            hs_pad: encoder output (B, T_in, d_model)
            hlens: encoder lengths (B,)
            ys_in_pad: decoder input token ids (B, T_out)
            ys_in_lens: decoder input lengths (B,)
            return_hs: also return hidden state before output layer

        Returns:
            x: logits (B, T_out, vocab_size)
            olens: output lengths (B,)
        """
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(ys_in_pad.device)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        tgt_mask = (tgt_mask & m).to(torch.uint8)

        src_mask = (~make_pad_mask(hlens, maxlen=hs_pad.size(1)))[:, None, :].to(hs_pad.device)

        x = self.dropout(
            self.tgt_word_emb(ys_in_pad) * self.scale
            + self.positional_encoding(ys_in_pad)
        )

        for dec_layer in self.layer_stack:
            x = dec_layer(x, hs_pad, tgt_mask, src_mask)

        x = self.layer_norm_out(x)

        if return_hs:
            hidden = x

        if self.tgt_word_prj is not None:
            x = self.tgt_word_prj(x)

        olens = tgt_mask.sum(1)
        if return_hs:
            return (x, hidden), olens
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        *,
        cache: Optional[List[Optional[Tensor]]] = None,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[Tensor]]:
        """One autoregressive step (beam search).

        Args:
            tgt: prefix token ids (B, T_out)
            tgt_mask: causal mask (B, T_out, T_out)
            memory: encoder output (B, T_in, d_model)
            memory_mask: encoder mask (B, 1, T_in)
            cache: per-layer KV cache
            return_hs: also return hidden state

        Returns:
            y: log-softmax scores (B, vocab_size)
            new_cache: updated cache list
        """
        x = self.dropout(
            self.tgt_word_emb(tgt) * self.scale
            + self.positional_encoding(tgt)
        )

        if cache is None:
            cache = [None] * self.n_layers

        new_cache: List[Tensor] = []
        for i, dec_layer in enumerate(self.layer_stack):
            x = dec_layer(x, memory, tgt_mask, memory_mask, cache=cache[i])
            new_cache.append(x)

        y = self.layer_norm_out(x[:, -1])

        if return_hs:
            hidden = y

        if self.tgt_word_prj is not None:
            y = torch.log_softmax(self.tgt_word_prj(y) / self.softmax_smoothing, dim=-1)

        if return_hs:
            return (y, hidden), new_cache
        return y, new_cache

    def score(self, ys, state, x, return_hs=False):
        """Single-sample beam search step."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if return_hs:
            (logp, hs), state = self.forward_one_step(
                ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state, return_hs=True
            )
            return logp.squeeze(0), hs, state
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: torch.Tensor,
        return_hs: bool = False,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Batched beam search scoring step."""
        n_batch = len(ys)
        n_layers = self.n_layers

        if states[0] is None:
            batch_state = None
        else:
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        if return_hs:
            (logp, hs), states = self.forward_one_step(
                ys, ys_mask, xs, cache=batch_state, return_hs=True
            )
        else:
            logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        if return_hs:
            return (logp, hs), state_list
        return logp, state_list

    def batch_beam_search(
        self,
        encoder_outputs: torch.Tensor,
        src_masks: torch.Tensor,
        beam_size: int = 1,
        nbest: int = 1,
        decode_max_len: int = 0,
        softmax_smoothing: float = 1.0,
        length_penalty: float = 0.0,
        eos_penalty: float = 1.0,
        elm=None,
        elm_weight: float = 0.0,
    ):
        if self.sos_id < 0 or self.eos_id < 0:
            raise ValueError("sos_id/eos_id must be set before batch_beam_search")

        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0

        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N * B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N * B, -1, Ti)
        ys = torch.ones(N * B, 1, device=device).fill_(self.sos_id).long()
        t_ys = ys.clone()
        confidences = torch.zeros(N * B, 1, device=device).float()
        caches: List[Optional[Tensor]] = [None for _ in range(self.n_layers)]
        scores = torch.tensor([0.0] + [-self.INF] * (B - 1), device=device).float()
        scores = scores.repeat(N).view(N * B, 1)
        is_finished = torch.zeros_like(scores)
        if elm is not None:
            elm_cache = elm.init_hidden(encoder_outputs, N * B)

        for _ in range(maxlen):
            tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)

            dec_output = self.dropout(
                self.tgt_word_emb(ys) * self.scale + self.positional_encoding(ys)
            )
            for i, dec_layer in enumerate(self.layer_stack):
                dec_output = dec_layer.forward(
                    dec_output,
                    encoder_outputs,
                    tgt_mask,
                    src_mask,
                    cache=caches[i],
                )
                caches[i] = dec_output

            dec_output = self.layer_norm_out(dec_output)
            t_logit = self.tgt_word_prj(dec_output[:, -1])
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)

            if elm is not None and elm_weight > 0.0:
                elm_logit, elm_cache = elm.forward_model(t_ys, hidden=elm_cache)
                t_lm_scores = torch.log_softmax(elm_logit[:, -1], dim=-1)
                t_lm_scores = t_lm_scores * (1 - is_finished.float())
                t_lm_scores[:, elm.eos_id] *= 3
                t_scores = t_scores + elm_weight * t_lm_scores

            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            scores = scores + t_topB_scores
            scores = scores.view(N, B * B)
            scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
            scores = scores.view(-1, 1)

            topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N * B)
            stride = B * torch.arange(N, device=device).view(N, 1).repeat(1, B).view(N * B)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

            ys = ys[topB_row_number_in_ys]
            t_ys = torch.gather(t_topB_ys.view(N, B * B), dim=1, index=topB_score_ids).view(N * B, 1)
            ys = torch.cat((ys, t_ys), dim=1)

            confidences = confidences[topB_row_number_in_ys]
            t_confidences = torch.gather(
                t_topB_scores.view(N, B * B), dim=1, index=topB_score_ids
            ).view(N * B, 1)
            confidences = torch.cat((confidences, torch.exp(t_confidences)), dim=1)

            new_caches: List[Optional[Tensor]] = []
            for cache in caches:
                if cache is not None:
                    new_caches.append(cache[topB_row_number_in_ys])
            caches = new_caches
            if elm is not None and elm_weight > 0.0:
                elm_cache = (
                    elm_cache[0][:, topB_row_number_in_ys],
                    elm_cache[1][:, topB_row_number_in_ys],
                )

            is_finished = t_ys.eq(self.eos_id)
            if is_finished.sum().item() == N * B:
                break

        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5 + ys_lengths.float()) / 6.0, length_penalty)
            scores = scores / penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        index = nbest_ids + B * torch.arange(N, device=device).view(N, 1).long()
        nbest_ys = ys.view(N * B, -1)[index.view(-1)].view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N * B)[index.view(-1)].view(N, -1)
        nbest_confidences = confidences.view(N * B, -1)[index.view(-1)].view(N, nbest_ids.size(1), -1)

        nbest_hyps = []
        for n in range(N):
            n_nbest_hyps = []
            for i, score in enumerate(nbest_scores[n]):
                confidence = nbest_confidences[n, i, 1:nbest_ys_lengths[n, i]].mean()
                n_nbest_hyps.append(
                    {
                        "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]],
                        "confidence": confidence,
                        "score": score,
                    }
                )
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    def ignored_target_position_is_0(self, padded_targets, ignore_id):
        mask = torch.ne(padded_targets, ignore_id)
        mask = mask.unsqueeze(dim=1)
        T = padded_targets.size(-1)
        upper_tri_0_mask = self.upper_triangular_is_0(T).unsqueeze(0).to(mask.dtype)
        upper_tri_0_mask = upper_tri_0_mask.to(mask.dtype).to(mask.device)
        return mask.to(torch.uint8) & upper_tri_0_mask.to(torch.uint8)

    def upper_triangular_is_0(self, size):
        ones = torch.ones(size, size)
        tri_left_ones = torch.tril(ones)
        return tri_left_ones.to(torch.uint8)

    def set_finished_beam_score_to_zero(self, scores, is_finished):
        NB, B = scores.size()
        is_finished = is_finished.float()
        mask_score = torch.tensor([0.0] + [-self.INF] * (B - 1), device=scores.device).float()
        mask_score = mask_score.view(1, B).repeat(NB, 1)
        return scores * (1 - is_finished) + mask_score * is_finished

    def set_finished_beam_y_to_eos(self, ys, is_finished):
        is_finished = is_finished.long()
        return ys * (1 - is_finished) + self.eos_id * is_finished

    def get_ys_lengths(self, ys):
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()



class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model * 4, dropout)

    def forward(self, dec_input, enc_output, self_attn_mask, cross_attn_mask,
                cache=None):
        x = dec_input
        residual = x
        x = self.self_attn_norm(x)
        if cache is not None:
            xq = x[:, -1:, :]
            residual = residual[:, -1:, :]
            self_attn_mask = self_attn_mask[:, -1:, :]
        else:
            xq = x
        x = self.self_attn(xq, x, x, mask=self_attn_mask)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, enc_output, enc_output, mask=cross_attn_mask)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)

        self.attention = DecoderScaledDotProductAttention(
            temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.fc(output)
        output = self.dropout(output)

        return output


class DecoderScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.INF = float("inf")

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.act(self.w_1(x)))
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item() / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)
        return self.pe[:, :length].clone().detach()
