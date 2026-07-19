"""Large GUI for the food-responsive bare neural Snake policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch

from snake_core import ACTION_FEATURE_NAMES, ACTION_NAMES, SnakeEnv, action_features
from train import DeepPolicyNet, load_policy_checkpoint


BASE_DIR = Path(__file__).resolve().parent
ARROWS = ("↑", "→", "↓", "←")
ACTION_COLORS = ("#38BDF8", "#A78BFA", "#FB923C", "#4ADE80")
BG, PANEL, TEXT, MUTED = "#07111F", "#0F172A", "#E5EEF9", "#8FA4BD"
CYAN, PINK, GOLD, FOOD = "#22D3EE", "#F472B6", "#FBBF24", "#FB7185"
SHORT_FEATURES = ("IN", "FREE", "≤FOOD", "<TAIL", "FWD", "CYCLE", "ΔFOOD", "EAT", "SAME")


class SnakeGUI:
    def __init__(self, root: tk.Tk, model_path: Path, seed: int | None = None,
                 cell: int = 16, target: int = 301, steps_per_frame: int = 100,
                 autoclose_frames: int = 0):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = load_policy_checkpoint(checkpoint)
        self.architecture = checkpoint.get("architecture", "linear_shared_action_scorer_v1")
        self.training_seed = int(checkpoint["seed"])
        self.seed = secrets.randbits(31) if seed is None else int(seed)
        self.env = SnakeEnv(50, self.seed)
        self.cell, self.target = cell, target
        self.steps_per_frame = steps_per_frame
        self.autoclose_frames = autoclose_frames
        self.frames = 0
        self.last_network_update = 0.0
        self.paused = False
        self.telemetry: dict | None = None
        self.last_frame_time = time.perf_counter()
        self.sps = 0.0
        self.root = root

        root.title("完整神经网络实时监视器 · 裸网络贪吃蛇 300+")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._style()
        self.status_var = tk.StringVar()
        tk.Label(root, textvariable=self.status_var, anchor="w",
                 font=("Microsoft YaHei UI", 15, "bold"), fg=TEXT, bg=PANEL,
                 padx=18, pady=11).pack(fill="x")
        body = tk.Frame(root, bg=BG)
        body.pack(padx=14, pady=12)
        pixels = 50 * cell
        self.board = tk.Canvas(body, width=pixels, height=pixels, bg="#06101C",
                               highlightthickness=1, highlightbackground="#243247")
        self.board.grid(row=0, column=0)
        side = tk.Frame(body, width=920, height=pixels, bg=PANEL)
        side.grid(row=0, column=1, padx=(14, 0))
        side.grid_propagate(False)
        tk.Label(side, text="FULL NETWORK TELEMETRY · DENSE BARE POLICY V3",
                 font=("Consolas", 16, "bold"), fg=CYAN, bg=PANEL).pack(
                     anchor="w", padx=16, pady=(14, 2))
        tk.Label(side, text="4 branches × [9 inputs → 9×128 live neurons → 1 raw logit] · shared weights",
                 font=("Consolas", 12), fg=MUTED, bg=PANEL).pack(anchor="w", padx=16)
        tk.Label(side, text="No action mask · No safety shield · model argmax executes directly",
                 font=("Consolas", 11, "bold"), fg=FOOD, bg=PANEL).pack(
                     anchor="w", padx=16, pady=(3, 8))
        self.net_canvas = tk.Canvas(side, width=888, height=545, bg="#0A1424",
                                    highlightthickness=1, highlightbackground="#243247")
        self.net_canvas.pack(padx=16)
        self._build_graph()
        self.output_var = tk.StringVar(value="Waiting for first food-responsive decision…")
        tk.Label(side, textvariable=self.output_var, justify="left", anchor="nw",
                 font=("Consolas", 11), fg=TEXT, bg=PANEL, padx=16, pady=8,
                 height=6).pack(fill="x")
        controls = tk.Frame(root, bg=BG)
        controls.pack(pady=(0, 13))
        for label, command in (("暂停 / 继续", self.toggle), ("单步", self.single_step),
                               ("重放同种子", self.replay), ("随机新局", self.random_game)):
            ttk.Button(controls, text=label, command=command).pack(side="left", padx=5)
        tk.Label(controls, text="速度", font=("Microsoft YaHei UI", 12, "bold"),
                 fg=TEXT, bg=BG).pack(side="left", padx=(18, 7))
        for label, value in (("1×", 1), ("20×", 20), ("100×", 100), ("500×", 500)):
            ttk.Button(controls, text=label,
                       command=lambda speed=value: self.set_speed(speed)).pack(side="left", padx=3)
        root.bind("<space>", lambda _event: self.toggle())
        root.bind("r", lambda _event: self.replay())
        self.tick()

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", font=("Microsoft YaHei UI", 12, "bold"),
                        padding=(12, 7), background="#1E293B", foreground=TEXT)
        style.map("TButton", background=[("active", "#334155")])

    def _build_graph(self) -> None:
        canvas = self.net_canvas
        self.feature_nodes: list[list[int]] = []
        self.activation_images: list[list[tk.PhotoImage]] = []
        self.activation_regions: list[dict] = []
        self.stage_edges: list[list[int]] = []
        self.output_nodes: list[int] = []
        self.output_texts: list[int] = []
        self.weight_images: list[tk.PhotoImage] = []
        self.weight_regions: list[dict] = []

        # Upper half: every live activation.  Each 16×8 micro-grid is one
        # complete 128-neuron layer; there is no sampling or hidden omission.
        canvas.create_text(14, 13, anchor="w", text="LIVE ACTIVATIONS · ALL 4,608 NEURONS",
                           fill=TEXT, font=("Consolas", 11, "bold"))
        canvas.create_text(872, 13, anchor="e",
                           text="dark = quiet  ·  bright = active  ·  gold = executed",
                           fill=MUTED, font=("Consolas", 9))
        canvas.create_text(62, 32, text="9 INPUTS", fill=MUTED,
                           font=("Consolas", 9, "bold"))
        layer_xs = [116 + 52 * index for index in range(9)]
        for index, x in enumerate(layer_xs):
            canvas.create_text(x + 15, 32, text=f"L{index + 1}", fill=MUTED,
                               font=("Consolas", 9, "bold"))
        canvas.create_text(606, 32, text="OUTPUT", fill=MUTED,
                           font=("Consolas", 9, "bold"))

        for action, centre_y in enumerate((59, 91, 123, 155)):
            canvas.create_text(17, centre_y, text=ARROWS[action],
                               fill=ACTION_COLORS[action],
                               font=("Microsoft YaHei UI", 16, "bold"))
            # Nine feature neurons as a compact 3×3 block.
            inputs: list[int] = []
            for feature in range(9):
                col, row = feature % 3, feature // 3
                x1, y1 = 48 + col * 8, centre_y - 11 + row * 8
                inputs.append(canvas.create_rectangle(
                    x1, y1, x1 + 6, y1 + 6, fill="#25364B", outline="",
                    tags=(f"feature:{action}:{feature}", "live-node")))
            self.feature_nodes.append(inputs)

            action_layers: list[tk.PhotoImage] = []
            action_edges: list[int] = []
            previous_x = 72
            for layer_index, x in enumerate(layer_xs):
                action_edges.append(canvas.create_line(
                    previous_x, centre_y, x - 3, centre_y,
                    fill="#29405A", width=2, tags="network-edge"))
                image = tk.PhotoImage(
                    data=self._activation_ppm(np.zeros(128, dtype=np.float32), 1.0),
                    format="PPM")
                canvas.create_image(x, centre_y - 8, anchor="nw", image=image,
                                    tags="activation-map")
                action_layers.append(image)
                self.activation_regions.append({
                    "x": x, "y": centre_y - 8, "width": 32, "height": 16,
                    "action": action, "layer": layer_index,
                })
                previous_x = x + 32
            action_edges.append(canvas.create_line(
                previous_x + 2, centre_y, 582, centre_y,
                fill="#29405A", width=2, tags="network-edge"))
            self.activation_images.append(action_layers)
            self.stage_edges.append(action_edges)
            self.output_nodes.append(canvas.create_oval(
                584, centre_y - 11, 628, centre_y + 11, fill="#172033",
                outline=ACTION_COLORS[action], width=2,
                tags=(f"output:{action}", "live-node")))
            self.output_texts.append(canvas.create_text(
                606, centre_y, text="0.0%", fill=TEXT,
                font=("Consolas", 9, "bold")))

        canvas.create_text(652, 48, anchor="nw",
                           text="LIVE STATE\nhover any neuron\nfor its exact value",
                           fill=MUTED, font=("Consolas", 9))
        self.hover_text = canvas.create_text(
            652, 108, anchor="nw", width=218,
            text="Move the pointer over a neuron or weight map.",
            fill=CYAN, font=("Consolas", 9, "bold"))

        canvas.create_line(14, 183, 874, 183, fill="#26384F")
        canvas.create_text(14, 199, anchor="w", text="ALL 132,352 TRAINED WEIGHTS",
                           fill=TEXT, font=("Consolas", 11, "bold"))
        canvas.create_text(872, 199, anchor="e",
                           text="every pixel = one connection weight",
                           fill=GOLD, font=("Consolas", 9, "bold"))

        # Lower half: lossless matrix heatmaps. Drawing 132k individual lines
        # would be both slower and less readable; one pixel per edge is exact.
        hidden_positions = [(18 + 136 * col, 224 + 146 * row)
                            for row in range(2) for col in range(4)]
        for index, (layer, (x, y)) in enumerate(
                zip(self.model.hidden_layers, hidden_positions), start=1):
            self._draw_weight_map(layer.weight, x, y, f"H{index} 128×128")
        self._draw_weight_map(self.model.input_layer.weight, 570, 224,
                              "INPUT 128×9", zoom=(4, 1))
        self._draw_weight_map(self.model.output_layer.weight, 570, 383,
                              "OUTPUT 1×128", zoom=(1, 10))

        canvas.create_text(730, 228, anchor="nw", text="WEIGHT DEPTH",
                           fill=TEXT, font=("Consolas", 10, "bold"))
        self._draw_weight_legend(730, 253)
        canvas.create_text(
            730, 327, anchor="nw", width=145,
            text="cyan  + positive\npink  − negative\ndark    near zero\nbright  large |w|",
            fill=MUTED, font=("Consolas", 9))
        canvas.create_text(
            570, 438, anchor="nw", width=300,
            text="Architecture: 9 → 128 → 8×128 → 1 · DENSE V3\n"
                 "The same 132,352 weights score all four actions.\n"
                 "No bias · ReLU after input and every hidden layer.",
            fill=MUTED, font=("Consolas", 9))
        canvas.bind("<Motion>", self._on_graph_motion)

    def _draw_weight_map(self, parameter: torch.Tensor, x: int, y: int,
                         label: str, zoom: tuple[int, int] = (1, 1)) -> None:
        matrix = parameter.detach().cpu().numpy().astype(np.float32)
        image = self._weight_photo(matrix)
        if zoom != (1, 1):
            image = image.zoom(*zoom)
        self.weight_images.append(image)
        self.net_canvas.create_text(x, y - 8, anchor="sw", text=label,
                                    fill=MUTED, font=("Consolas", 8, "bold"))
        self.net_canvas.create_image(x, y, anchor="nw", image=image,
                                     tags="weight-map")
        height, width = matrix.shape
        self.weight_regions.append({
            "x": x, "y": y, "width": width * zoom[0],
            "height": height * zoom[1], "zoom": zoom,
            "matrix": matrix, "label": label.split()[0],
        })

    @staticmethod
    def _weight_photo(matrix: np.ndarray) -> tk.PhotoImage:
        magnitude = np.abs(matrix)
        nonzero = magnitude[magnitude > 0]
        # A robust per-matrix scale keeps small but real dense connections
        # visible instead of letting one large value black out the whole map.
        scale = max(float(np.percentile(nonzero, 99.0)) if nonzero.size else 0.0, 1e-8)
        strength = np.sqrt(np.clip(magnitude / scale, 0.0, 1.0))[..., None]
        dark = np.array((9, 20, 35), dtype=np.float32)
        positive = np.array((34, 211, 238), dtype=np.float32)
        negative = np.array((244, 114, 182), dtype=np.float32)
        target = np.where((matrix >= 0)[..., None], positive, negative)
        rgb = (dark + (target - dark) * strength).astype(np.uint8)
        height, width = matrix.shape
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
        return tk.PhotoImage(data=ppm, format="PPM")

    @staticmethod
    def _activation_ppm(values: np.ndarray, scale: float) -> bytes:
        values = np.asarray(values, dtype=np.float32).reshape(8, 16)
        strength = np.sqrt(np.clip(np.abs(values) / max(scale, 1e-8), 0.0, 1.0))
        dark = np.array((13, 27, 44), dtype=np.float32)
        active = np.array((34, 211, 238), dtype=np.float32)
        rgb = dark + (active - dark) * strength[..., None]
        # Each of the 128 neurons is a stable 2×2 pixel cell.  Updating 36
        # images replaces 4,608 Tcl Canvas mutations per telemetry frame.
        rgb = np.repeat(np.repeat(rgb.astype(np.uint8), 2, axis=0), 2, axis=1)
        return b"P6\n32 16\n255\n" + rgb.tobytes()

    def _draw_weight_legend(self, x: int, y: int) -> None:
        for index in range(9):
            value = -1.0 + index / 4.0
            self.net_canvas.create_rectangle(
                x + index * 15, y, x + index * 15 + 14, y + 14,
                fill=self._color(value, 1.0), outline="")
        self.net_canvas.create_text(x, y + 20, anchor="nw", text="−P99     0      +P99",
                                    fill=MUTED, font=("Consolas", 8))

    def _on_graph_motion(self, event: tk.Event) -> None:
        current = self.net_canvas.find_withtag("current")
        if current:
            tags = self.net_canvas.gettags(current[0])
            for tag in tags:
                parts = tag.split(":")
                if parts[0] == "feature" and self.telemetry is not None:
                    action, feature = map(int, parts[1:])
                    value = float(self.telemetry["matrix"][action, feature])
                    self.net_canvas.itemconfigure(
                        self.hover_text,
                        text=f"{ARROWS[action]} input[{feature}]\n{SHORT_FEATURES[feature]} = {value:+.5f}")
                    return
                if parts[0] == "output" and self.telemetry is not None:
                    action = int(parts[1])
                    self.net_canvas.itemconfigure(
                        self.hover_text,
                        text=(f"{ARROWS[action]} {ACTION_NAMES[action]} output\n"
                              f"logit = {self.telemetry['logits'][action]:+.6f}\n"
                              f"probability = {self.telemetry['probabilities'][action]:.3%}"))
                    return
        if self.telemetry is not None:
            for region in self.activation_regions:
                if (region["x"] <= event.x < region["x"] + region["width"] and
                        region["y"] <= event.y < region["y"] + region["height"]):
                    column = (event.x - region["x"]) // 2
                    row = (event.y - region["y"]) // 2
                    neuron = row * 16 + column
                    action, layer = region["action"], region["layer"]
                    value = float(self.telemetry["layers"][layer][action, neuron])
                    self.net_canvas.itemconfigure(
                        self.hover_text,
                        text=f"{ARROWS[action]} L{layer + 1} neuron {neuron}\nactivation = {value:+.6f}")
                    return
        for region in self.weight_regions:
            if (region["x"] <= event.x < region["x"] + region["width"] and
                    region["y"] <= event.y < region["y"] + region["height"]):
                zx, zy = region["zoom"]
                column = (event.x - region["x"]) // zx
                row = (event.y - region["y"]) // zy
                value = float(region["matrix"][row, column])
                self.net_canvas.itemconfigure(
                    self.hover_text,
                    text=f"{region['label']} weight [{row}, {column}]\nw = {value:+.7f}")
                return

    @staticmethod
    def _color(value: float, scale: float = 1.0) -> str:
        strength = min(1.0, abs(value) / max(scale, 1e-6))
        base = np.array((34, 211, 238) if value >= 0 else (244, 114, 182))
        dark = np.array((30, 41, 59))
        rgb = (dark + (base - dark) * strength).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    @torch.inference_mode()
    def forward_snapshot(self) -> tuple[int, dict]:
        matrix = action_features(self.env)
        tensor = torch.from_numpy(matrix)
        logits_tensor, network_state = self.model.forward_with_telemetry(tensor)
        logits = logits_tensor.numpy().copy()
        probabilities = torch.softmax(logits_tensor, dim=0).numpy().copy()
        action = int(np.argmax(logits))
        # Food counterfactual: same snake, food mirrored across board center.
        mirror = (49 - self.env.food[0], 49 - self.env.food[1])
        offset = 0
        while mirror in self.env.occupied:
            offset += 1
            mirror = ((49 - self.env.food[0] + 7 * offset) % 50,
                      (49 - self.env.food[1] + 11 * offset) % 50)
        alternate = self.model(torch.from_numpy(action_features(self.env, mirror))).numpy()
        return action, {
            "matrix": matrix.copy(),
            "contributions": network_state["contributions"].numpy().copy(),
            "layers": [layer.numpy().copy() for layer in network_state["layers"]],
            "logits": logits, "probabilities": probabilities, "action": action,
            "alternate_action": int(np.argmax(alternate)), "mirror_food": mirror,
            "state_step": self.env.steps, "head": self.env.head, "food": self.env.food,
        }

    def _execute_batch(self, count: int) -> int:
        executed = 0
        for index in range(count):
            # Length 301 is a score milestone, not an episode terminator.
            # Keep playing until collision or the entire 50x50 board is full.
            if not self.env.alive or self.env.length >= self.env.size * self.env.size:
                self.paused = True
                break
            action, telemetry = self.forward_snapshot()
            self.telemetry = telemetry
            self.env.step(action)  # raw argmax only
            executed += 1
        return executed

    def update_network(self) -> None:
        data = self.telemetry
        if data is None:
            return
        matrix = data["matrix"]
        for action in range(4):
            for feature in range(9):
                value = float(matrix[action, feature])
                self.net_canvas.itemconfigure(self.feature_nodes[action][feature],
                                              fill=self._color(value, 1.0))
            for layer_index, layer in enumerate(data["layers"]):
                activation = layer[action]
                # Per-layer normalization retains useful depth when adjacent
                # layers have very different numeric ranges.
                scale = max(float(np.percentile(np.abs(activation), 95)), 1e-6)
                self.activation_images[action][layer_index].configure(
                    data=self._activation_ppm(activation, scale), format="PPM")
            chosen = action == data["action"]
            for edge in self.stage_edges[action]:
                self.net_canvas.itemconfigure(
                    edge, fill=GOLD if chosen else "#29405A",
                    width=3 if chosen else 2)
            self.net_canvas.itemconfigure(self.output_nodes[action],
                                          outline=GOLD if chosen else ACTION_COLORS[action],
                                          width=4 if chosen else 2,
                                          fill="#332A16" if chosen else "#172033")
            self.net_canvas.itemconfigure(
                self.output_texts[action],
                text=f"{float(data['probabilities'][action]):.1%}",
                fill=GOLD if chosen else TEXT)
        lines = []
        for action in range(4):
            gain = matrix[action, 6]
            direction = "CLOSER" if gain > 0 else ("FARTHER" if gain < 0 else "SAME")
            lines.append(f"{ARROWS[action]} {ACTION_NAMES[action]:5s} food gain={gain:+.0f} {direction:7s} "
                         f"logit={data['logits'][action]:+7.3f}  p={data['probabilities'][action]:6.2%}")
        closer_probability = float(data["probabilities"][matrix[:, 6] > 0].sum())
        changed = data["alternate_action"] != data["action"]
        lines.extend((
            f"RAW ARGMAX → {ARROWS[data['action']]} {ACTION_NAMES[data['action']]} → env.step({data['action']})",
            f"P(closer to food)={closer_probability:.1%} · food-mirror preference changed={changed}",
        ))
        self.output_var.set("\n".join(lines))

    def draw_board(self) -> None:
        self.board.delete("dynamic")
        c = self.cell
        hr, hc = self.env.head
        fr, fc = self.env.food
        self.board.create_line(hc*c+c/2, hr*c+c/2, fc*c+c/2, fr*c+c/2,
                               fill="#7F3D56", dash=(5, 5), width=1, tags="dynamic")
        self.board.create_oval(fc*c+2, fr*c+2, (fc+1)*c-2, (fr+1)*c-2,
                               fill=FOOD, outline="", tags="dynamic")
        for index, (row, column) in enumerate(self.env.snake):
            color = GOLD if index == 0 else "#22C55E"
            pad = 1 if index == 0 else 2
            self.board.create_rectangle(column*c+pad, row*c+pad, (column+1)*c-pad,
                                        (row+1)*c-pad, fill=color, outline="", tags="dynamic")
        state = "暂停" if self.paused else ("运行中" if self.env.alive else "碰撞死亡")
        if self.env.length >= self.env.size * self.env.size:
            reached = "  ✓ 已填满整个棋盘"
        elif self.env.length >= self.target:
            reached = "  ✓ 已超过 300，继续运行中"
        else:
            reached = ""
        self.status_var.set(
            f"{state}   长度 {self.env.length}/301   食物 {self.env.foods}   步数 {self.env.steps:,}   "
            f"{self.sps:,.0f} SPS   {self.steps_per_frame}×   游戏 seed {self.seed}   "
            f"训练 seed {self.training_seed}{reached}"
        )

    def tick(self) -> None:
        now = time.perf_counter()
        elapsed = max(now - self.last_frame_time, 1e-6)
        executed = 0 if self.paused else self._execute_batch(self.steps_per_frame)
        self.sps = executed / elapsed
        self.last_frame_time = now
        self.draw_board()
        # Batched telemetry images refresh at 10 FPS while the board remains at
        # 30 FPS. This is both smoother and much safer for Tk on Windows.
        if now - self.last_network_update >= 0.1:
            self.update_network()
            self.last_network_update = now
        self.frames += 1
        if self.autoclose_frames and self.frames >= self.autoclose_frames:
            self.root.destroy()
            return
        self.root.after(33, self.tick)

    def toggle(self) -> None: self.paused = not self.paused
    def set_speed(self, speed: int) -> None: self.steps_per_frame = speed

    def single_step(self) -> None:
        self.paused = True
        self._execute_batch(1)
        self.draw_board(); self.update_network()

    def _reset(self, seed: int) -> None:
        self.seed = int(seed); self.env = SnakeEnv(50, self.seed)
        self.paused = False; self.telemetry = None; self.sps = 0.0

    def replay(self) -> None: self._reset(self.seed)
    def random_game(self) -> None: self._reset(secrets.randbits(31))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=BASE_DIR / "models" / "snake_policy.pt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cell", type=int, default=16)
    parser.add_argument("--target", type=int, default=301)
    parser.add_argument("--speed", type=int, default=100, choices=(1, 20, 100, 500))
    parser.add_argument("--autoclose-frames", type=int, default=0)
    args = parser.parse_args()
    root = tk.Tk()
    SnakeGUI(root, args.model, args.seed, args.cell, args.target, args.speed,
             args.autoclose_frames)
    root.mainloop()


if __name__ == "__main__":
    main()
