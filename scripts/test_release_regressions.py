"""CPU regressions for release fixes; no ASR backend, Mamba, or audio download.

Run from any directory with Python and the CPU test dependencies installed:
    python scripts/test_release_regressions.py

The tests execute the actual source functions/classes. AST extraction isolates
unavailable model/toolkit imports; small torch modules replace only the heavy
CleanMel/STFT components. The bundled mask helper stands in for ESPnet's helper
when exercising the overlay frontends. This does not test full GPU inference.
"""

import ast
import builtins
import logging
import math
from pathlib import Path
import sys
import tempfile
import types
from typing import Tuple
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.utils as nn_utils


ROOT = Path(__file__).resolve().parents[1]
DEMO = "funasr_prop_demo/funasr_src/"
FRONTENDS = (
    DEMO + "frontend/cleanmel_funasr.py",
    "src/espnet2/asr/frontend/cleanmel_funasr.py",
    "src/espnet2/asr/frontend/cleanmel_fireredasr.py",
)


def source_symbols(relative_path, names, extra=None):
    """Execute unchanged selected definitions, without importing heavy deps."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing requested source definitions in {path}")
    namespace = {
        "torch": torch, "nn": nn, "Tensor": torch.Tensor,
        "Tuple": Tuple, "nn_utils": nn_utils, "math": math,
        "logging": logging, "AbsFrontend": nn.Module,
    }
    namespace.update(extra or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


MASK = source_symbols(
    DEMO + "frontend/nets_utils.py",
    {"make_pad_mask", "_make_pad_mask", "_make_pad_mask_traceable", "triu_onnx"},
)["make_pad_mask"]
SOXNORM = source_symbols(DEMO + "frontend/stft.py", {"soxnorm"})["soxnorm"]


class ConstructorTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class SmallCleanMel(nn.Module):
    """Only the model weights are replaced when testing real loader code."""
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        self.bias = nn.Parameter(torch.zeros(2))


def frontend_class(relative_path):
    return source_symbols(relative_path, {"CleanMelFrontend"}, {
        "CleanMel": SmallCleanMel,
        "FunasrInputSTFT": ConstructorTransform,
        "FunasrTargetMel": ConstructorTransform,
        "make_pad_mask": MASK,
    })["CleanMelFrontend"]


class StaticSTFT(nn.Module):
    def forward(self, wav):
        batch, samples = wav.shape
        frames = (samples - 512) // 256 + 1
        return torch.zeros(batch, 257, frames, 2), torch.ones(batch, 1)


class StaticMel(nn.Module):
    def forward(self, wav, scale):
        batch, samples = wav.shape
        frames = (samples - 512) // 256 + 1
        # Distinct, nonzero Mel values expose accidental frequency-axis masking.
        return torch.arange(1, 81).float().view(1, 80, 1).expand(batch, 80, frames)


class ZeroMaskLogits(nn.Module):
    def forward(self, stft):
        return torch.zeros(stft.shape[0], 80, stft.shape[2])


class FrontendMaskTests(unittest.TestCase):
    def run_frontend(self, path, samples, lengths):
        cls = frontend_class(path)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.stft = StaticSTFT()
        model.target_mel = StaticMel()
        model.cleanmel = ZeroMaskLogits()
        return model(torch.zeros(len(lengths), samples), torch.tensor(lengths))

    def test_one_second_preserves_all_eighty_mel_bins(self):
        expected_frame = (torch.arange(1, 81).float() * 0.25 * (1 << 30)).log()
        for path in FRONTENDS:
            with self.subTest(frontend=path):
                mel, lengths, *_ = self.run_frontend(path, 16000, [16000])
                self.assertEqual(tuple(mel.shape), (1, 61, 80))
                self.assertEqual(lengths.tolist(), [61])
                torch.testing.assert_close(mel, expected_frame.view(1, 1, 80).expand_as(mel))

    def test_batch_masks_only_time_padding(self):
        expected_frame = (torch.arange(1, 81).float() * 0.25 * (1 << 30)).log()
        for path in FRONTENDS:
            with self.subTest(frontend=path):
                mel, lengths, *_ = self.run_frontend(path, 32000, [16000, 32000])
                self.assertEqual(lengths.tolist(), [61, 124])
                torch.testing.assert_close(mel[0, :61], expected_frame.expand(61, 80))
                torch.testing.assert_close(mel[0, 61:], torch.full_like(mel[0, 61:], math.log(1e6)))
                torch.testing.assert_close(mel[1], expected_frame.expand(124, 80))


class NormalizationTests(unittest.TestCase):
    def test_silence_and_reused_factor_are_finite(self):
        for shape in [(16000,), (1, 16000), (2, 16000)]:
            with self.subTest(shape=shape):
                wav = torch.zeros(shape)
                actual, factor = SOXNORM(wav, -3)
                self.assertTrue(math.isfinite(factor))
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, wav, rtol=0, atol=0)
                reused, reused_factor = SOXNORM(wav, None, factor)
                self.assertEqual(factor, reused_factor)
                torch.testing.assert_close(reused, wav, rtol=0, atol=0)

    def test_nonsilent_scaling_preserves_original_numerics(self):
        wav = torch.tensor([[0.0, 0.1, -0.7], [0.3, -0.2, 0.05]])
        expected_factor = 10 ** (-3 / 20) / wav.abs().max().item()
        actual, factor = SOXNORM(wav, -3)
        self.assertEqual(factor, expected_factor)
        torch.testing.assert_close(actual, wav * expected_factor, rtol=0, atol=0)
        reused, _ = SOXNORM(wav, None, factor)
        torch.testing.assert_close(reused, actual, rtol=0, atol=0)

    def test_mixed_silent_and_nonsilent_batch_keeps_old_global_scale(self):
        wav = torch.tensor([[0.0, 0.0], [-0.5, 0.2]])
        expected_factor = 10 ** (-3 / 20) / 0.5
        actual, factor = SOXNORM(wav, -3)
        self.assertEqual(factor, expected_factor)
        torch.testing.assert_close(actual, wav * expected_factor, rtol=0, atol=0)


class NoTkImportTests(unittest.TestCase):
    def test_actual_demo_frontend_module_import_does_not_require_tk(self):
        # Execute the full source module, with optional GPU/ASR deps isolated.
        names = ["transformers", "transformers.generation", "librosa", "frontend",
                 "frontend.stft", "frontend.nets_utils", "mamba_ssm",
                 "mamba_ssm.utils", "mamba_ssm.utils.generation"]
        modules = {name: types.ModuleType(name) for name in names}
        for module in modules.values():
            module.__path__ = []
        modules["transformers"].generation = modules["transformers.generation"]
        modules["transformers.generation"].GreedySearchDecoderOnlyOutput = object
        modules["frontend.stft"].FunasrInputSTFT = ConstructorTransform
        modules["frontend.stft"].FunasrTargetMel = ConstructorTransform
        modules["frontend.nets_utils"].make_pad_mask = MASK
        modules["mamba_ssm"].Mamba = nn.Module
        modules["mamba_ssm.utils.generation"].InferenceParams = object
        original_import = builtins.__import__

        def without_tk(name, *args, **kwargs):
            if name.split(".")[0] in {"tkinter", "_tkinter"}:
                raise ModuleNotFoundError("Tk intentionally unavailable for regression")
            return original_import(name, *args, **kwargs)

        path = ROOT / FRONTENDS[0]
        namespace = {"__name__": "release_frontend_without_tk", "__file__": str(path)}
        with patch.dict(sys.modules, modules), patch("builtins.__import__", without_tk):
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        self.assertIn("CleanMelFrontend", namespace)


class CleanMelCheckpointTests(unittest.TestCase):
    def test_complete_native_pretraining_and_adapted_formats(self):
        weights = {"weight": torch.arange(6).float().reshape(2, 3), "bias": torch.tensor([3., 4.])}
        for path in FRONTENDS:
            for prefix in ["", "arch.", "frontend.cleanmel."]:
                for wrapped in [False, True]:
                    with self.subTest(frontend=path, prefix=prefix, wrapped=wrapped):
                        state = {prefix + k: v for k, v in weights.items()}
                        if prefix:
                            state["preencoder.unrelated"] = torch.ones(1)
                        if wrapped:
                            state = {"state_dict": state}
                        with tempfile.TemporaryDirectory() as tmp:
                            checkpoint = Path(tmp) / "complete.pth"
                            torch.save(state, checkpoint)
                            model = frontend_class(path)(model_path=str(checkpoint))
                        for name, expected in weights.items():
                            torch.testing.assert_close(model.cleanmel.state_dict()[name], expected, rtol=0, atol=0)

    def test_missing_unexpected_and_wrong_shape_fail(self):
        for path in FRONTENDS:
            for defect in ["missing", "unexpected", "wrong_shape"]:
                with self.subTest(frontend=path, defect=defect):
                    state = {"frontend.cleanmel.weight": torch.ones(2, 3),
                             "frontend.cleanmel.bias": torch.ones(2)}
                    if defect == "missing":
                        del state["frontend.cleanmel.bias"]
                    elif defect == "unexpected":
                        state["frontend.cleanmel.extra"] = torch.zeros(1)
                    else:
                        state["frontend.cleanmel.weight"] = torch.ones(3, 2)
                    with tempfile.TemporaryDirectory() as tmp:
                        checkpoint = Path(tmp) / "incomplete.pth"
                        torch.save(state, checkpoint)
                        with self.assertRaisesRegex(RuntimeError, "Cannot load complete CleanMel weights"):
                            frontend_class(path)(model_path=str(checkpoint))


class AdapterCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = source_symbols(DEMO + "frontend/linear.py", {"BatchedLFR", "LinearProjection"})["LinearProjection"]
        cls.checkpoint = ROOT / "checkpoints/funasr_prop_frontend.pth"
        state = torch.load(cls.checkpoint, map_location="cpu", weights_only=True)
        cls.weights = {k: v for k, v in state.items() if k.startswith("preencoder.")}

    def test_shipped_checkpoint_restores_every_adapter_tensor(self):
        model = self.adapter(80, 560, ckpt=str(self.checkpoint))
        actual = model.state_dict()
        self.assertEqual(len(actual), len(self.weights))
        for key, expected in self.weights.items():
            torch.testing.assert_close(actual[key[len("preencoder."):]], expected, rtol=0, atol=0)

    def test_missing_unexpected_and_wrong_shape_fail(self):
        for defect in ["missing", "unexpected", "wrong_shape"]:
            with self.subTest(defect=defect):
                state = dict(self.weights)
                if defect == "missing":
                    del state["preencoder.out.bias"]
                elif defect == "unexpected":
                    state["preencoder.extra"] = torch.ones(1)
                else:
                    state["preencoder.out.bias"] = torch.ones(81)
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = Path(tmp) / "incomplete.pth"
                    torch.save(state, checkpoint)
                    with self.assertRaisesRegex(RuntimeError, "Cannot load complete Adapter weights"):
                        self.adapter(80, 560, ckpt=str(checkpoint))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
