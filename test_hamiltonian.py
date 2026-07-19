import unittest
from pathlib import Path

from snake_core import DIRECTIONS, hamiltonian_cycle, hamiltonian_targets


class HamiltonianPolicyTest(unittest.TestCase):
    def test_cycle_covers_every_cell_and_closes(self):
        cycle = hamiltonian_cycle(50)
        self.assertEqual(len(cycle), 2500)
        self.assertEqual(len(set(cycle)), 2500)
        for current, nxt in zip(cycle, cycle[1:] + cycle[:1]):
            self.assertEqual(abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]), 1)

    def test_cycle_matches_initial_snake_direction(self):
        targets = hamiltonian_targets(50)
        for point in ((25, 23), (25, 24), (25, 25)):
            self.assertEqual(targets[point[0] * 50 + point[1]], 1)  # RIGHT

    def test_runtime_has_no_action_mask_or_safety_shield(self):
        combined = "\n".join(Path(name).read_text(encoding="utf-8")
                             for name in ("train.py", "evaluate.py", "gui.py"))
        self.assertNotIn("safe_actions", combined)
        self.assertNotIn("safety_analysis", combined)
        self.assertNotIn("masked_action", combined)
        self.assertNotIn(".legal_actions(", combined)


if __name__ == "__main__":
    unittest.main()
