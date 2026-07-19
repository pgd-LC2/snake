"""Snake rules and the training-only Hamiltonian-cycle teacher."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import random
import secrets

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")
ACTION_FEATURE_NAMES = (
    "inside", "body_free", "before_food", "before_tail", "forward",
    "cycle_progress", "food_gain", "is_food", "same_direction",
)
# Training-teacher score. These weights are targets for supervised distillation,
# never an output-time correction or fallback.
TEACHER_WEIGHTS = np.asarray((1.0, 1.0, 1.0, 1.0, 1.0,
                              0.02, 0.20, 0.50, 0.001), dtype=np.float32)


@dataclass
class StepResult:
    alive: bool
    ate: bool


class SnakeEnv:
    def __init__(self, size: int = 50, seed: int | None = None):
        self.size = size
        self.seed = secrets.randbits(63) if seed is None else int(seed)
        self.rng = random.Random(self.seed)
        self.snake: deque[tuple[int, int]] = deque()
        self.occupied: set[tuple[int, int]] = set()
        self.direction = 1
        self.food = (0, 0)
        self.alive = True
        self.steps = 0
        self.foods = 0
        self.reset()

    @property
    def length(self) -> int:
        return len(self.snake)

    @property
    def head(self) -> tuple[int, int]:
        return self.snake[0]

    def reset(self) -> None:
        c = self.size // 2
        self.snake = deque([(c, c), (c, c - 1), (c, c - 2)])
        self.occupied = set(self.snake)
        self.direction = 1
        self.alive = True
        self.steps = 0
        self.foods = 0
        self._place_food()

    def _place_food(self) -> None:
        if len(self.occupied) == self.size * self.size:
            self.food = (-1, -1)
            return
        while True:
            point = (self.rng.randrange(self.size), self.rng.randrange(self.size))
            if point not in self.occupied:
                self.food = point
                return

    def next_head(self, action: int) -> tuple[int, int]:
        dr, dc = DIRECTIONS[action]
        r, c = self.head
        return r + dr, c + dc

    def legal_actions(self) -> list[int]:
        """Diagnostics only. The naked neural policy does not call this method."""
        legal: list[int] = []
        tail = self.snake[-1]
        for action in range(4):
            nr, nc = self.next_head(action)
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            if (nr, nc) in self.occupied and (nr, nc) != tail:
                continue
            legal.append(action)
        return legal

    def step(self, action: int) -> StepResult:
        """Execute exactly the supplied action; no mask, shield, or correction."""
        if not self.alive:
            return StepResult(False, False)
        new_head = self.next_head(int(action))
        ate = new_head == self.food
        tail = self.snake[-1]
        inside = 0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size
        hit_body = new_head in self.occupied and not (new_head == tail and not ate)
        if not inside or hit_body:
            self.alive = False
            return StepResult(False, False)
        self.snake.appendleft(new_head)
        self.occupied.add(new_head)
        if ate:
            self.foods += 1
            self._place_food()
        else:
            old_tail = self.snake.pop()
            if old_tail != new_head:
                self.occupied.remove(old_tail)
        self.direction = int(action)
        self.steps += 1
        return StepResult(True, ate)


def hamiltonian_cycle(size: int = 50) -> list[tuple[int, int]]:
    """Return a cycle whose direction matches the initial snake (moving right)."""
    if size < 2 or size % 2:
        raise ValueError("This construction requires an even board size >= 2.")
    forward: list[tuple[int, int]] = [(0, c) for c in range(size)]
    for row in range(1, size):
        columns = range(size - 1, 0, -1) if row % 2 else range(1, size)
        forward.extend((row, column) for column in columns)
    forward.extend((row, 0) for row in range(size - 1, 0, -1))
    return list(reversed(forward))


def hamiltonian_targets(size: int = 50) -> list[int]:
    """Training labels indexed by flattened head cell."""
    cycle = hamiltonian_cycle(size)
    targets = [0] * (size * size)
    for index, current in enumerate(cycle):
        nxt = cycle[(index + 1) % len(cycle)]
        delta = (nxt[0] - current[0], nxt[1] - current[1])
        targets[current[0] * size + current[1]] = DIRECTIONS.index(delta)
    return targets


@lru_cache(maxsize=4)
def cycle_position_table(size: int = 50) -> tuple[int, ...]:
    positions = [0] * (size * size)
    for index, (row, column) in enumerate(hamiltonian_cycle(size)):
        positions[row * size + column] = index
    return tuple(positions)


def action_features(env: SnakeEnv, food_override: tuple[int, int] | None = None) -> np.ndarray:
    """Raw per-action observations consumed by the deployed neural network.

    This function does not choose, mask, replace, or rank actions. Collision and
    ordering signals are inputs, analogous to sensors; model argmax remains final.
    """
    size = env.size
    total = size * size
    positions = cycle_position_table(size)
    food = env.food if food_override is None else food_override
    head_cycle = positions[env.head[0] * size + env.head[1]]
    food_cycle = (positions[food[0] * size + food[1]] - head_cycle) % total
    tail = env.snake[-1]
    tail_cycle = (positions[tail[0] * size + tail[1]] - head_cycle) % total
    current_manhattan = abs(food[0] - env.head[0]) + abs(food[1] - env.head[1])
    rows: list[list[float]] = []
    for action, (dr, dc) in enumerate(DIRECTIONS):
        point = (env.head[0] + dr, env.head[1] + dc)
        inside = 0 <= point[0] < size and 0 <= point[1] < size
        body_free = inside and not (point in env.occupied and point != tail)
        progress = ((positions[point[0] * size + point[1]] - head_cycle) % total
                    if inside else 0)
        next_manhattan = (abs(food[0] - point[0]) + abs(food[1] - point[1])
                          if inside else current_manhattan + 1)
        rows.append([
            float(inside),
            float(body_free),
            float(inside and 0 < progress <= food_cycle),
            float(inside and 0 < progress < tail_cycle),
            float(inside and progress > 0),
            progress / total,
            float(current_manhattan - next_manhattan),
            float(inside and point == food),
            float(action == env.direction),
        ])
    return np.asarray(rows, dtype=np.float32)


def teacher_scores(feature_matrix: np.ndarray) -> np.ndarray:
    """Food-driven shortcut teacher used for labels only."""
    return np.asarray(feature_matrix, dtype=np.float32) @ TEACHER_WEIGHTS


def teacher_action(env: SnakeEnv) -> int:
    return int(np.argmax(teacher_scores(action_features(env))))
