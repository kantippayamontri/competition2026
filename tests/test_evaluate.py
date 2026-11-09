import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch


CLASSMAP = {"anoDR": 0, "bmildDR": 1, "cmoderateDR": 2, "dsevereDR": 3, "eproDR": 4}
VALIDATION = Path("idrid_resplit_seed2026/idrid_resplit_seed2026/val")


class EvaluateCLITest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "evaluator"
        self.base.mkdir()
        source = Path(__file__).resolve().parents[1] / "evaluate.py"
        shutil.copy2(source, self.base / "evaluate.py")
        (self.base / "class_to_idx.txt").write_text(json.dumps(CLASSMAP), encoding="utf-8")
        self.cwd = self.root / "unrelated"
        self.cwd.mkdir()
        self.model_path = self.root / "toy.pt"
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 5)
        ).eval()
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.zero_()
            model[-1].bias[2] = 1
        torch.jit.trace(model, torch.zeros(1, 3, 512, 512)).save(str(self.model_path))

    def add_image(self, dataset, label):
        directory = self.base / dataset / label
        directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(40, 80, 120)).save(directory / "sample.png")

    def run_evaluator(self, mode=None):
        command = [sys.executable, str(self.base / "evaluate.py"), str(self.model_path)]
        if mode is not None:
            command.append(mode)
        return subprocess.run(
            command,
            cwd=self.cwd,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            text=True, capture_output=True, timeout=60,
        )

    def assert_score(self, result, total, correct, accuracy):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         {"total": total, "correct": correct, "accuracy": accuracy})

    def assert_no_score(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_hidden_only_subset_uses_script_relative_path(self):
        self.add_image("test_idrid", "anoDR")
        self.add_image("test_idrid", "cmoderateDR")
        for mode in ("hidden", "anything-except-training"):
            with self.subTest(mode=mode):
                self.assert_score(self.run_evaluator(mode), 2, 1, 50.0)

    def test_validation_only_supports_training_and_default_mode(self):
        for label in CLASSMAP:
            self.add_image(VALIDATION, label)
        for mode in ("training", None):
            with self.subTest(mode=mode):
                self.assert_score(self.run_evaluator(mode), 5, 1, 20.0)

    def test_missing_hidden_does_not_use_validation(self):
        for label in CLASSMAP:
            self.add_image(VALIDATION, label)
        self.assert_no_score(self.run_evaluator("hidden"))

    def test_missing_validation_does_not_use_hidden(self):
        self.add_image("test_idrid", "cmoderateDR")
        self.assert_no_score(self.run_evaluator("training"))

    def test_empty_or_unmapped_hidden_does_not_emit_zero_score(self):
        (self.base / "test_idrid").mkdir()
        with self.subTest(contents="empty"):
            self.assert_no_score(self.run_evaluator("hidden"))
        self.add_image("test_idrid", "unmapped")
        with self.subTest(contents="unmapped"):
            self.assert_no_score(self.run_evaluator("hidden"))

    def test_hidden_skips_unmapped_folders_without_stdout_noise(self):
        self.add_image("test_idrid", "anoDR")
        self.add_image("test_idrid", "cmoderateDR")
        self.add_image("test_idrid", "unmapped")
        self.assert_score(self.run_evaluator("hidden"), 2, 1, 50.0)


if __name__ == "__main__":
    unittest.main()
