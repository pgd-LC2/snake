"""Anti-cheat evaluation for the food-responsive bare neural policy."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import secrets
import statistics

import numpy as np
import torch

from snake_core import SnakeEnv, action_features
from train import load_policy_checkpoint, rollout_cycle


BASE_DIR = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate_batch(model: torch.nn.Module, seeds: list[int], target: int):
    envs = [SnakeEnv(50, seed) for seed in seeds]
    action_sets = [dict() for _ in envs]
    mismatch_counts = [0 for _ in envs]
    counterfactual_features: list[np.ndarray] = []
    while True:
        active = [i for i, env in enumerate(envs)
                  if env.alive and env.length < target and env.steps < 120_000]
        if not active:
            break
        matrices = np.stack([action_features(envs[i]) for i in active])
        logits = model(torch.from_numpy(matrices)).numpy()
        actions = logits.argmax(1)
        for local_index, env_index in enumerate(active):
            env = envs[env_index]
            expected_argmax = int(np.argmax(logits[local_index]))
            action = int(actions[local_index])
            mismatch_counts[env_index] += int(action != expected_argmax)
            cell = env.head[0] * 50 + env.head[1]
            action_sets[env_index].setdefault(cell, set()).add(action)
            if len(counterfactual_features) < 500 and env.steps % 700 == 0:
                candidates = ((1, 1), (1, 48), (48, 48), (48, 1))
                valid_foods = [point for point in candidates if point not in env.occupied]
                if len(valid_foods) >= 3:
                    counterfactual_features.append(
                        np.stack([action_features(env, food_override=point)
                                  for point in valid_foods[:4]]))
            env.step(action)  # exactly model argmax; no mask, shield, or fallback
    rows = []
    global_actions: dict[int, set[int]] = {}
    for seed, env, per_cell, mismatches in zip(seeds, envs, action_sets, mismatch_counts):
        for cell, actions in per_cell.items():
            global_actions.setdefault(cell, set()).update(actions)
        rows.append({
            "seed": seed, "length": env.length, "foods": env.foods,
            "steps": env.steps, "alive": env.alive,
            "reached_target": env.length >= target,
            "action_mismatches": mismatches,
            "multi_successor_cells": sum(len(actions) > 1 for actions in per_cell.values()),
            "visited_cells": len(per_cell),
        })
    diversity = sum(len(actions) > 1 for actions in global_actions.values()) / len(global_actions)
    responsive = 0
    changed_logits = 0
    if counterfactual_features:
        for alternatives in counterfactual_features:
            output = model(torch.from_numpy(alternatives)).numpy()
            changed_logits += int(float(np.max(np.abs(output - output[0]))) > 1e-6)
            responsive += int(len(set(output.argmax(1).tolist())) >= 2)
    cf_count = len(counterfactual_features)
    return rows, {
        "counterfactual_snapshots": cf_count,
        "food_changes_logits_rate": changed_logits / cf_count if cf_count else 0.0,
        "food_changes_argmax_rate": responsive / cf_count if cf_count else 0.0,
        "multi_successor_cell_rate": diversity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=BASE_DIR / "models" / "snake_policy.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--target", type=int, default=301)
    parser.add_argument("--seed", type=int, default=None,
                        help="master evaluation seed; omitted means OS-random")
    parser.add_argument("--json", type=Path,
                        default=BASE_DIR / "models" / "evaluation_300.json")
    parser.add_argument("--csv", type=Path,
                        default=BASE_DIR / "models" / "evaluation_300_episodes.csv")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(31)
    rng = random.Random(args.seed)
    seeds = list(dict.fromkeys(rng.randrange(1, 2**31) for _ in range(args.episodes * 2)))
    seeds = seeds[:args.episodes]

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = load_policy_checkpoint(checkpoint)
    rows, behavior = evaluate_batch(model, seeds, args.target)
    baseline_steps = [rollout_cycle(seed, args.target) for seed in seeds]
    raw_steps = [int(row["steps"]) for row in rows]
    ratios = [raw / baseline for raw, baseline in zip(raw_steps, baseline_steps)]
    successes = sum(bool(row["reached_target"]) for row in rows)
    summary = {
        "agent_mode": "food_responsive_bare_nn_argmax_no_mask_no_shield",
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "training_seed": int(checkpoint["seed"]),
        "evaluation_master_seed": args.seed,
        "evaluation_seeds": seeds,
        "episodes": args.episodes,
        "target_length": args.target,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "median_steps": statistics.median(raw_steps),
        "p95_steps": float(np.percentile(raw_steps, 95)),
        "median_hamiltonian_baseline_steps": statistics.median(baseline_steps),
        "median_step_ratio_vs_fixed_sweep": statistics.median(ratios),
        "paired_wins_vs_fixed_sweep": sum(raw < base for raw, base in zip(raw_steps, baseline_steps)),
        "action_mismatches": sum(int(row["action_mismatches"]) for row in rows),
        **behavior,
        "episodes_detail": rows,
        "baseline_steps": baseline_steps,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    printable = {key: value for key, value in summary.items()
                 if key not in ("episodes_detail", "baseline_steps", "evaluation_seeds")}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    passed = (
        successes == args.episodes
        and summary["action_mismatches"] == 0
        and summary["median_step_ratio_vs_fixed_sweep"] < 0.25
        and summary["food_changes_logits_rate"] >= 0.95
        and summary["food_changes_argmax_rate"] >= 0.30
        and summary["multi_successor_cell_rate"] >= 0.25
    )
    if not passed:
        raise SystemExit("Food-responsive bare-policy anti-cheat evaluation failed.")


if __name__ == "__main__":
    main()
