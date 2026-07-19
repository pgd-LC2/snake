import unittest

from snake_core import SnakeEnv


class SnakeRulesTest(unittest.TestCase):
    def test_seed_is_reproducible(self):
        self.assertEqual(SnakeEnv(50, 7).food, SnakeEnv(50, 7).food)

    def test_food_never_starts_on_snake(self):
        for seed in range(100):
            env = SnakeEnv(50, seed)
            self.assertNotIn(env.food, env.occupied)

    def test_wall_collision(self):
        env = SnakeEnv(50, 1)
        for _ in range(30):
            result = env.step(0)
            if not result.alive:
                break
        self.assertFalse(env.alive)

    def test_eating_grows_by_one(self):
        env = SnakeEnv(50, 1)
        env.food = env.next_head(1)
        before = env.length
        result = env.step(1)
        self.assertTrue(result.ate)
        self.assertEqual(env.length, before + 1)
        self.assertNotIn(env.food, env.occupied)


if __name__ == "__main__":
    unittest.main()
