import logging
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.linear_decoder import LinearDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.legacy.nets.e2e_asr_common import ErrorCalculator
from espnet2.legacy.nets.pytorch_backend.nets_utils import th_accuracy
from espnet2.legacy.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.legacy.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

autocast_type = torch.float16
if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast

    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        autocast_type = torch.bfloat16
else:
    @contextmanager
    def autocast(enabled=True):
        yield


class FireRedASR2SModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: torch.nn.Module,
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: torch.nn.Module,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        aux_ctc: Optional[dict] = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        autocast_frontend: bool = False,
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        adptor_weight_start: float = 1.0,
        adptor_weight_end: float = 1.0,
        adptor_weight_schedule_steps: int = 1,
        adptor_target_source: str = "noisy",
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight < 1.0, interctc_weight

        super().__init__()
        sym_blank = token_list[-1]
        if sym_blank in token_list:
            self.blank_id = token_list.index(sym_blank)
        else:
            self.blank_id = vocab_size - 1
        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight = interctc_weight
        self.aux_ctc = aux_ctc
        self.token_list = token_list.copy()

        self.frontend = frontend
        from espnet2.asr.frontend.wav_frontend import WavFrontend
        self.org_frontend = WavFrontend(
            fs=16000,
            window="hamming",
            n_mels=80,
            frame_length=25,
            frame_shift=10,
            filter_length_min=-1,
            filter_length_max=-1,
            lfr_m=1,
            lfr_n=1,
            cmvn_file=None,
            dither=0.0,
            snip_edges=True,
            upsacle_samples=True,
            cmvn=None,
        )

        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.autocast_frontend = autocast_frontend

        if self.encoder and getattr(self.encoder, "interctc_use_conditioning", False):
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.use_linear_decoder = isinstance(decoder, LinearDecoder)

        self.error_calculator = ErrorCalculator(
            token_list, sym_space, sym_blank, report_cer, report_wer
        )

        if ctc_weight < 1.0:
            assert (
                decoder is not None
            ), "decoder should not be None when attention is used"
        else:
            decoder = None
            logging.warning("Set decoder to none as ctc_weight==1.0")

        self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        if self.encoder is not None:
            self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__
        else:
            self.is_encoder_whisper = False

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None

        self.adptor_weight_start = adptor_weight_start
        self.adptor_weight_end = adptor_weight_end
        self.adptor_weight_schedule_steps = adptor_weight_schedule_steps
        if adptor_target_source not in ("noisy", "clean"):
            raise ValueError(
                "adptor_target_source must be one of {'noisy', 'clean'}"
            )
        self.adptor_target_source = adptor_target_source
        self.training_step_count = 0

        model_path = os.environ.get(
            "FIRERED_MODEL_PATH", "checkpoints/fireredasr/model.pth.tar"
        )
        state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage, weights_only=False
        )["model_state_dict"]
        encoder_state_dict = {}
        ctc_state_dict = {}
        decoder_state_dict = {}
        for key in list(state_dict.keys()):
            if "encoder" in key:
                encoder_state_dict[key.replace("encoder.", "")] = state_dict[key]
            elif "ctc" in key:
                ctc_state_dict[key.replace("ctc.", "")] = state_dict[key]
            elif "decoder" in key:
                decoder_state_dict[key.replace("decoder.", "")] = state_dict[key]
        
        model_dict = self.encoder.state_dict()
        model_dict.update(encoder_state_dict)
        self.encoder.load_state_dict(model_dict)

        if self.ctc is not None:
            self.ctc.load_state_dict(ctc_state_dict)

        if self.decoder is not None:
            model_dict = self.decoder.state_dict()
            model_dict.update(decoder_state_dict)
            self.decoder.load_state_dict(model_dict)

        


        





    def _get_adptor_weight(self, step: Optional[int] = None) -> float:
        """
        Return the current loss_adptor weight (linearly annealed).
        
        Parameters:
            step: optional step index; if omitted, the internal counter is used.
                  The weight anneals from adptor_weight_start to adptor_weight_end.
                  
        Returns:
            float: current weight in [adptor_weight_end, adptor_weight_start].
        """
        if step is None:
            step = self.training_step_count
        
        progress = min(step / self.adptor_weight_schedule_steps, 1.0)
        weight = (self.adptor_weight_start - self.adptor_weight_end) * (1 - progress) + self.adptor_weight_end
        return weight
    
    def set_adptor_weight_schedule(
        self, 
        start_weight: float = 1.0, 
        end_weight: float = 0.1,
        schedule_steps: int = 100000
    ) -> None:
        """Configure the linear annealing schedule for loss_adptor."""
        self.adptor_weight_start = start_weight
        self.adptor_weight_end = end_weight
        self.adptor_weight_schedule_steps = schedule_steps

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        text = text[:, : text_lengths.max()]

        encoder_out, encoder_out_lens, feats, feats_lengths, x_norm = self.encode(speech, speech_lengths)
        target_speech = speech
        target_speech_lengths = speech_lengths
        if self.adptor_target_source == "clean":
            target_speech = kwargs.get("speech_clean", speech)
            target_speech_lengths = kwargs.get("speech_clean_lengths", speech_lengths)

        target_speech_norm = target_speech * x_norm
        org_feats, org_lengths = self.org_frontend(
            target_speech_norm, target_speech_lengths
        )

        
        if feats.shape[1] <= org_feats.shape[1]:
            org_feats = org_feats[:, :feats.shape[1], :]
            org_lengths = feats_lengths
        else:
            feats = feats[:, :org_feats.shape[1], :]
            feats_lengths = org_lengths
        
        from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
        mask_feats = (make_pad_mask(feats_lengths).unsqueeze(-1)).to(feats.device)
        feats = feats.masked_fill(mask_feats, 0.0) 
        org_feats = org_feats.masked_fill(mask_feats, 0.0)
        loss_adptor = torch.nn.functional.mse_loss(feats, org_feats, reduction='sum') / (feats_lengths.sum() * org_feats.shape[-1])
   

        
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        loss_ctc, cer_ctc = None, None
        stats = dict()
        stats["loss_adptor"] = loss_adptor.detach()



        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out,
                encoder_out_lens,
                text,
                text_lengths,
                utt_ids=kwargs.get("utt_id"),
            )

            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        loss_interctc = 0.0
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            for layer_idx, intermediate_out in intermediate_outs:

                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                encoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc = loss_interctc + loss_ic

                stats["loss_interctc_layer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

            loss_interctc = loss_interctc / len(intermediate_outs)

            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        if self.ctc_weight != 1.0:
            loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att
        
        current_step = kwargs.get("step", None)
        adptor_weight = self._get_adptor_weight(step=current_step)
        
  
        if adptor_weight > 0.0:
            loss = adptor_weight * loss_adptor + loss
        else:
            loss = loss
        
        stats["adptor_weight"] = adptor_weight
        
        if current_step is None:
            self.training_step_count += 1

        stats["loss"] = loss.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        feats, feats_lengths, x_norm = self._extract_feats(speech, speech_lengths)



        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)
        

        adaptor_feats, adaptor_feats_lengths = feats.clone(), feats_lengths.clone()
        if self.encoder is None:
            encoder_out, encoder_out_lens = feats, feats_lengths
        else:

            if getattr(self.encoder, "interctc_use_conditioning", False) or getattr(
                self.encoder, "ctc_trim", False
            ):
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc
                )
            else:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if self.encoder is not None and (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens, adaptor_feats, adaptor_feats_lengths, x_norm


    def encode_cleanmel(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        feats, feats_lengths = speech, speech_lengths
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        if self.encoder is None:
            encoder_out, encoder_out_lens = feats, feats_lengths
        else:

            if getattr(self.encoder, "interctc_use_conditioning", False) or getattr(
                self.encoder, "ctc_trim", False
            ):
                encoder_out, encoder_out_lens, _ = self.encoder(
                    feats, feats_lengths, ctc=self.ctc
                )
            else:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if self.encoder is not None and (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens
    

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        speech = speech[:, : speech_lengths.max()]

        x_norm = None
        if self.frontend is not None:
            frontend_out = self.frontend(speech, speech_lengths)
            if len(frontend_out) == 3:
                feats, feats_lengths, x_norm = frontend_out
            else:
                feats, feats_lengths = frontend_out
        else:
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths, x_norm


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        utt_ids: Optional[List[str]] = None,
    ):
        loss_ctc = self.ctc(
            encoder_out,
            encoder_out_lens,
            ys_pad,
            ys_pad_lens,
            utt_ids=utt_ids,
        )

        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc


    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        text = text[:, : text_lengths.max()]

        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc

    def _calc_classif_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute classification loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, 1)
        Return:
            loss_classif: Classification loss value.
            acc_classif: Classification accuracy.
        """
        assert labels.dim() == 2, labels.shape
        assert labels.shape[1] == 1, labels.shape
        logits = self.decoder(encoder_out, encoder_out_lens)
        assert logits.shape[1] == self.vocab_size - 3, logits.shape
        labels = labels - 2
        loss_classif = self.criterion_classif(logits, labels.squeeze(-1))
        acc_classif = th_accuracy(logits, labels, ignore_label=self.ignore_id)
        return loss_classif, acc_classif
