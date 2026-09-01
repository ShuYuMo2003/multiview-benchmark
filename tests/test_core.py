from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvbench.checkpoint import load_checkpoint, save_checkpoint
from mvbench.contracts import PairBatch, PairMetricOutput
from mvbench.data import LeRobotPairDataset, LeRobotSchema, MockPairDataset, PairPlan
from mvbench.geometry import quaternion_to_matrix, relative_rotation_log
from mvbench.losses import MultitaskConsistencyLoss
from mvbench.metrics import EvaluationAccumulator
from mvbench.models import PairwiseStateConsistencyModel


class CorePipelineTest(unittest.TestCase):
    def _model(self) -> PairwiseStateConsistencyModel:
        return PairwiseStateConsistencyModel.from_config({
            "backbone": {"name": "tiny", "output_dim": 32, "patch_size": 7},
            "fusion": {"dim": 64, "depth": 2, "heads": 4, "mlp_ratio": 2.0},
            "joint_residual_dim": 7,
            "dino_residual_dim": 16,
        })

    def _batch(self) -> PairBatch:
        dataset = MockPairDataset(
            episodes=2, frames_per_episode=8, image_size=28, joint_dim=7, dino_dim=16
        )
        mapping = next(iter(DataLoader(dataset, batch_size=6, shuffle=False)))
        batch = PairBatch.from_mapping(mapping)
        batch.validate()
        return batch

    def test_model_loss_backward(self) -> None:
        batch = self._batch()
        model = self._model()
        output = model(batch.head_image, batch.wrist_image, batch.side)
        self.assertEqual(output.translation_residual.shape, (6, 3))
        self.assertEqual(output.rotation_residual.shape, (6, 3))
        self.assertEqual(output.joint_residual.shape, (6, 7))
        self.assertEqual(output.dino_residual.shape, (6, 16))
        loss, diagnostics = MultitaskConsistencyLoss()(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss/dino_vector", diagnostics)
        loss.backward()
        self.assertIsNotNone(model.dino_head.weight.grad)
        self.assertIsNotNone(model.validity_head.weight.grad)

    def test_perfect_metrics(self) -> None:
        batch = self._batch()
        energy = torch.where(
            batch.consistency_label > 0.5,
            torch.full_like(batch.consistency_label, -10.0),
            torch.full_like(batch.consistency_label, 10.0),
        )
        output = PairMetricOutput(
            translation_residual=batch.translation_residual.clone(),
            rotation_residual=batch.rotation_residual.clone(),
            joint_residual=batch.joint_residual.clone(),
            gripper_residual=batch.gripper_residual.clone(),
            dino_residual=batch.dino_residual.clone(),
            compatibility_energy=energy,
            validity_logit=torch.zeros_like(energy),
        )
        accumulator = EvaluationAccumulator()
        accumulator.update(output, batch)
        metrics = accumulator.compute(distributed=False)
        self.assertAlmostEqual(metrics["translation/error_mean"], 0.0)
        self.assertAlmostEqual(metrics["joint/error_mean"], 0.0)
        self.assertAlmostEqual(metrics["dino/error_mean"], 0.0)
        self.assertAlmostEqual(metrics["energy/auroc"], 1.0)

    def test_checkpoint_roundtrip(self) -> None:
        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, None, None, 2, 17, {"test": True})
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            payload = load_checkpoint(path, model, optimizer, restore_rng=False)
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["global_step"], 17)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]))


class GeometryAndAdapterTest(unittest.TestCase):
    def test_relative_quaternion_rotation(self) -> None:
        identity = quaternion_to_matrix(torch.tensor([0.0, 0.0, 0.0, 1.0]), "xyzw")
        quarter_turn = quaternion_to_matrix(
            torch.tensor([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]), "xyzw"
        )
        log = relative_rotation_log(identity, quarter_turn)
        self.assertTrue(torch.allclose(log, torch.tensor([0.0, 0.0, np.pi / 2]), atol=1e-5))

    def test_lerobot_row_adapter(self) -> None:
        def row(offset: float) -> dict[str, torch.Tensor]:
            return {
                "observation.images.front_head": torch.zeros(3, 20, 30),
                "observation.images.left_wrist": torch.ones(3, 10, 20),
                "observation.images.right_wrist": torch.ones(3, 10, 20) * 0.5,
                "left_pos": torch.tensor([offset, 0.0, 0.0]),
                "right_pos": torch.tensor([0.0, offset, 0.0]),
                "left_rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "right_rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "left_joint": torch.arange(7).float() + offset,
                "right_joint": torch.arange(7).float() - offset,
                "left_grip": torch.tensor([offset]),
                "right_grip": torch.tensor([offset]),
                "left_dino": torch.arange(4).float() + offset,
                "right_dino": torch.arange(4).float() - offset,
            }

        plan = PairPlan(
            anchor_index=np.asarray([1]), candidate_index=np.asarray([0]),
            side=np.asarray([0]), consistency_label=np.asarray([0.0]),
            same_episode=np.asarray([True]),
        )
        schema = LeRobotSchema(
            head_image_key="observation.images.front_head",
            left_wrist_image_key="observation.images.left_wrist",
            right_wrist_image_key="observation.images.right_wrist",
            left_eef_position_key="left_pos", right_eef_position_key="right_pos",
            left_eef_rotation_key="left_rot", right_eef_rotation_key="right_rot",
            left_joint_key="left_joint", right_joint_key="right_joint",
            left_gripper_key="left_grip", right_gripper_key="right_grip",
            left_dino_key="left_dino", right_dino_key="right_dino",
            image_size=28,
        )
        dataset = LeRobotPairDataset([row(0.0), row(0.25)], plan, schema, joint_dim=7, dino_dim=4)
        sample = dataset[0]
        self.assertEqual(sample["head_image"].shape, (3, 28, 28))
        self.assertTrue(torch.allclose(sample["translation_residual"], torch.tensor([0.25, 0.0, 0.0])))
        self.assertTrue(torch.allclose(sample["joint_residual"], torch.full((7,), 0.25)))
        self.assertTrue(torch.allclose(sample["dino_residual"], torch.full((4,), 0.25)))


if __name__ == "__main__":
    unittest.main()
