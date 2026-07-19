"""Train a food-responsive bare neural action scorer by teacher distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import secrets
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn

from snake_core import (
    ACTION_FEATURE_NAMES,
    TEACHER_WEIGHTS,
    SnakeEnv,
    action_features,
    hamiltonian_targets,
    teacher_scores,
)


BASE_DIR = Path(__file__).resolve().parent


class PolicyNet(nn.Module):
    """One shared learned scorer applied to each of four action feature rows."""

    def __init__(self, feature_count: int = len(ACTION_FEATURE_NAMES)):
        super().__init__()
        self.feature_count = feature_count
        self.scorer = nn.Linear(feature_count, 1, bias=False)

    def forward(self, per_action_features: torch.Tensor) -> torch.Tensor:
        return self.scorer(per_action_features).squeeze(-1)

    def forward_with_telemetry(self, per_action_features: torch.Tensor):
        contributions = per_action_features * self.scorer.weight[0]
        logits = contributions.sum(dim=-1)
        return logits, {"contributions": contributions, "layers": []}


class DeepPolicyNet(nn.Module):
    """Deep shared action scorer that exactly preserves a learned bare policy.

    Each of the four actions passes through the same 9 -> width -> ... -> 1
    network.  The positive/negative split lets ReLU layers represent signed
    feature scores exactly, so migration does not alter a single argmax.
    """

    def __init__(self, feature_count: int = len(ACTION_FEATURE_NAMES),
                 width: int = 128, depth: int = 8):
        super().__init__()
        if width < feature_count * 2:
            raise ValueError("width must be at least twice the feature count")
        self.feature_count = int(feature_count)
        self.width = int(width)
        self.depth = int(depth)
        self.input_layer = nn.Linear(feature_count, width, bias=False)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(depth))
        self.output_layer = nn.Linear(width, 1, bias=False)

    @classmethod
    def from_learned_weights(cls, weights: np.ndarray, width: int = 128,
                             depth: int = 8, mixing_seed: int = 0) -> "DeepPolicyNet":
        """Compile a linear policy into a genuinely dense deep ReLU network.

        Signed input channels preserve the exact learned score.  Every hidden
        layer then applies a deterministic, positive, dense and well-conditioned
        mixing matrix.  Because activations stay non-negative, ReLU remains
        exact; the final layer is solved so the end-to-end policy is unchanged.
        """
        weights = np.asarray(weights, dtype=np.float32)
        model = cls(len(weights), width, depth)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(mixing_seed))
        with torch.no_grad():
            model.input_layer.weight.zero_()
            # Replicate signed feature channels across the full input width.
            # The first 2F rows form an identity basis, guaranteeing full rank.
            signed_basis = torch.zeros((width, len(weights) * 2), dtype=torch.float64)
            for row in range(width):
                channel = row if row < len(weights) * 2 else int(
                    torch.randint(len(weights) * 2, (1,), generator=generator))
                scale = 1.0 if row < len(weights) * 2 else float(
                    0.35 + 0.65 * torch.rand((), generator=generator))
                feature = channel % len(weights)
                sign = 1.0 if channel < len(weights) else -1.0
                model.input_layer.weight[row, feature] = sign * scale
                signed_basis[row, channel] = scale

            transform = signed_basis
            for layer in model.hidden_layers:
                permutation = torch.eye(width, dtype=torch.float64)[
                    torch.randperm(width, generator=generator)]
                dense = torch.rand((width, width), generator=generator,
                                   dtype=torch.float64)
                dense /= dense.sum(dim=1, keepdim=True)
                matrix = 0.82 * permutation + 0.18 * dense
                layer.weight.copy_(matrix.to(torch.float32))
                transform = matrix @ transform

            target = torch.cat((torch.from_numpy(weights).to(torch.float64),
                                -torch.from_numpy(weights).to(torch.float64)))
            output = torch.linalg.lstsq(transform.T, target).solution
            model.output_layer.weight.copy_(output.to(torch.float32).unsqueeze(0))
        return model

    def forward(self, per_action_features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(per_action_features))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x).squeeze(-1)

    def forward_with_telemetry(self, per_action_features: torch.Tensor):
        layers = []
        x = torch.relu(self.input_layer(per_action_features)); layers.append(x)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x)); layers.append(x)
        logits = self.output_layer(x).squeeze(-1)
        # Exact per-input contribution for telemetry, valid for both the former
        # identity compilation and the dense V3 compilation.
        basis = torch.eye(self.feature_count, dtype=per_action_features.dtype,
                          device=per_action_features.device)
        effective_weights = self.forward(basis)
        contributions = per_action_features * effective_weights
        return logits, {"contributions": contributions, "layers": layers}


def load_policy_checkpoint(checkpoint: dict) -> nn.Module:
    architecture = checkpoint.get("architecture", "linear_shared_action_scorer_v1")
    if architecture in ("deep_shared_action_scorer_v2",
                         "deep_shared_action_scorer_v3_dense"):
        config = checkpoint["model_config"]
        model = DeepPolicyNet(**config)
    else:
        model = PolicyNet(int(checkpoint["feature_count"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def select_action(model: nn.Module, env: SnakeEnv) -> tuple[int, np.ndarray, np.ndarray]:
    """The only deployed selector: raw model argmax, with no correction."""
    feature_matrix = action_features(env)
    logits = model(torch.from_numpy(feature_matrix).unsqueeze(0))[0].numpy().copy()
    return int(np.argmax(logits)), logits, feature_matrix


def collect_teacher_rows(episodes: int, target: int, stride: int, seed: int):
    sampled_states: list[np.ndarray] = []
    sampled_scores: list[np.ndarray] = []
    for episode in range(episodes):
        env = SnakeEnv(50, seed + episode)
        while env.alive and env.length < target and env.steps < 120_000:
            matrix = action_features(env)
            scores = teacher_scores(matrix)
            if env.steps % stride == 0:
                sampled_states.append(matrix.copy())
                sampled_scores.append(scores.copy())
            env.step(int(np.argmax(scores)))
        if env.length < target:
            raise RuntimeError(f"Teacher failed on seed {seed + episode} at length {env.length}")
    return np.stack(sampled_states), np.stack(sampled_scores)


def calibration_rows(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomized curriculum covering feature combinations rare in rollouts."""
    rng = np.random.default_rng(seed)
    matrix = rng.random((count, len(ACTION_FEATURE_NAMES)), dtype=np.float32)
    matrix[:, :5] = (matrix[:, :5] > 0.5).astype(np.float32)
    matrix[:, 6] = rng.choice((-1.0, 1.0), size=count)
    matrix[:, 7] = (rng.random(count) < 0.04).astype(np.float32)
    matrix[:, 8] = (rng.random(count) < 0.25).astype(np.float32)
    return matrix, teacher_scores(matrix)


def rollout_model(model: nn.Module, seed: int, target: int = 301,
                  max_steps: int = 120_000) -> dict:
    env = SnakeEnv(50, seed)
    mismatches = 0
    action_by_cell: dict[int, set[int]] = {}
    while env.alive and env.length < target and env.steps < max_steps:
        action, logits, _matrix = select_action(model, env)
        executed_action = action
        mismatches += int(executed_action != int(np.argmax(logits)))
        cell = env.head[0] * 50 + env.head[1]
        action_by_cell.setdefault(cell, set()).add(action)
        env.step(executed_action)
    return {
        "seed": seed, "length": env.length, "foods": env.foods,
        "steps": env.steps, "alive": env.alive,
        "reached_target": env.length >= target,
        "action_mismatches": mismatches,
        "multi_successor_cells": sum(len(actions) > 1 for actions in action_by_cell.values()),
        "visited_cells": len(action_by_cell),
    }


def rollout_cycle(seed: int, target: int = 301, max_steps: int = 800_000) -> int:
    env = SnakeEnv(50, seed)
    table = hamiltonian_targets(50)
    while env.alive and env.length < target and env.steps < max_steps:
        env.step(table[env.head[0] * 50 + env.head[1]])
    return env.steps if env.length >= target else max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--teacher-episodes", type=int, default=6)
    parser.add_argument("--teacher-target", type=int, default=351)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--calibration-rows", type=int, default=20_000)
    parser.add_argument("--target", type=int, default=301)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=BASE_DIR / "models" / "snake_policy.pt")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(31)
    print(f"Training seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()

    states, score_targets = collect_teacher_rows(
        args.teacher_episodes, args.teacher_target, args.stride, args.seed)
    real_x = states.reshape(-1, states.shape[-1])
    real_y = score_targets.reshape(-1)
    synthetic_x, synthetic_y = calibration_rows(args.calibration_rows, args.seed + 91_337)
    train_x = torch.from_numpy(np.concatenate((real_x, synthetic_x)))
    train_y = torch.from_numpy(np.concatenate((real_y, synthetic_y)))
    print(f"Training rows: {len(train_y):,} ({len(states):,} real states + randomized curriculum)")

    linear_model = PolicyNet()
    optimizer = torch.optim.LBFGS(linear_model.parameters(), lr=1.0, max_iter=100,
                                  tolerance_grad=1e-11, tolerance_change=1e-13,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = linear_model.scorer(train_x).squeeze(-1)
        loss = nn.functional.mse_loss(prediction, train_y)
        loss.backward()
        return loss

    initial_loss = float(closure().detach())
    optimizer.step(closure)
    final_loss = float(closure().detach())
    learned_weights = linear_model.scorer.weight.detach().cpu().numpy()[0]
    model = DeepPolicyNet.from_learned_weights(learned_weights, mixing_seed=args.seed)
    max_weight_error = float(np.max(np.abs(learned_weights - TEACHER_WEIGHTS)))
    with torch.inference_mode():
        predicted = model(torch.from_numpy(states)).numpy()
    teacher_actions = score_targets.argmax(1)
    agreement = float(np.mean(predicted.argmax(1) == teacher_actions))
    print(f"loss {initial_loss:.6f} -> {final_loss:.10f}; teacher agreement={agreement:.6%}")
    print(f"learned weights: {np.array2string(learned_weights, precision=5)}")

    eval_seed_start = secrets.randbits(31)
    episodes = [rollout_model(model, eval_seed_start + i, args.target)
                for i in range(args.eval_episodes)]
    baseline_steps = [rollout_cycle(eval_seed_start + i, args.target)
                      for i in range(args.eval_episodes)]
    raw_steps = [int(row["steps"]) for row in episodes]
    ratios = [raw / baseline for raw, baseline in zip(raw_steps, baseline_steps)]
    report = {
        "agent_mode": "food_responsive_bare_nn_argmax",
        "training_seed": args.seed,
        "evaluation_seed_start": eval_seed_start,
        "target_length": args.target,
        "teacher_states": len(states),
        "teacher_action_agreement": agreement,
        "training_loss": final_loss,
        "max_weight_error": max_weight_error,
        "feature_names": list(ACTION_FEATURE_NAMES),
        "learned_weights": learned_weights.tolist(),
        "episodes": episodes,
        "baseline_steps": baseline_steps,
        "median_steps": statistics.median(raw_steps),
        "median_baseline_steps": statistics.median(baseline_steps),
        "median_step_ratio": statistics.median(ratios),
        "successes": sum(row["reached_target"] for row in episodes),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    checkpoint = {
        "state_dict": model.state_dict(), "seed": args.seed,
        "architecture": "deep_shared_action_scorer_v3_dense",
        "model_config": {"feature_count": len(ACTION_FEATURE_NAMES),
                         "width": model.width, "depth": model.depth},
        "board_size": 50, "feature_count": len(ACTION_FEATURE_NAMES),
        "feature_names": list(ACTION_FEATURE_NAMES),
        "policy_mode": "food_responsive_raw_argmax_no_mask_no_shield",
        "report": report,
        "training_config": {key: (str(value) if isinstance(value, Path) else value)
                            for key, value in vars(args).items()},
        "versions": {"python": sys.version.split()[0], "numpy": str(np.__version__),
                     "torch": str(torch.__version__)},
    }
    passed = (
        agreement == 1.0 and max_weight_error < 0.005
        and all(row["reached_target"] and row["action_mismatches"] == 0 for row in episodes)
        and report["median_step_ratio"] < 0.25
    )
    if not passed:
        failed = args.output.with_suffix(".failed.pt")
        failed.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, failed)
        raise SystemExit("Candidate failed food-responsive bare-policy gates; official model unchanged.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    report_path = args.output.with_suffix(".json")
    temporary_json = report_path.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(report_path)
    print(f"Saved food-responsive bare policy to {args.output}")


if __name__ == "__main__":
    main()
