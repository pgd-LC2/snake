import unittest
from collections import deque
from pathlib import Path

import torch

from snake_core import SnakeEnv, action_features
from train import load_policy_checkpoint, select_action


class FoodResponsivePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).resolve().parent / "models" / "snake_policy.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        cls.checkpoint = checkpoint
        cls.model = load_policy_checkpoint(checkpoint)

    def test_same_snake_changes_action_when_food_moves(self):
        env = SnakeEnv(50, 123)
        actions = set()
        with torch.inference_mode():
            for food in ((1, 1), (1, 48), (48, 48), (48, 1)):
                logits = self.model(torch.from_numpy(action_features(env, food)))
                actions.add(int(logits.argmax()))
        self.assertGreaterEqual(len(actions), 2)

    def test_checkpoint_declares_raw_policy(self):
        model_path = Path(__file__).resolve().parent / "models" / "snake_policy.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["policy_mode"],
                         "food_responsive_raw_argmax_no_mask_no_shield")

    def test_hidden_layers_are_dense_not_identity_compilation(self):
        self.assertEqual(self.checkpoint["architecture"],
                         "deep_shared_action_scorer_v3_dense")
        identity = torch.eye(self.model.width)
        for layer in self.model.hidden_layers:
            self.assertEqual(torch.count_nonzero(layer.weight).item(),
                             layer.weight.numel())
            self.assertFalse(torch.allclose(layer.weight, identity))

    def test_illegal_argmax_is_not_corrected(self):
        class AlwaysUp(torch.nn.Module):
            def forward(self, feature_matrix):
                batch_shape = feature_matrix.shape[:-2]
                logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
                return logits.expand(*batch_shape, 4)

        env = SnakeEnv(50, 7)
        env.snake = deque([(0, 10), (1, 10), (2, 10)])
        env.occupied = set(env.snake)
        action, logits, _features = select_action(AlwaysUp(), env)
        self.assertEqual(action, 0)
        self.assertEqual(action, int(logits.argmax()))
        result = env.step(action)
        self.assertFalse(result.alive)


if __name__ == "__main__":
    unittest.main()
