# 50×50 稠密裸神经网络贪吃蛇（最终 V3）

这是清理后的核心版本。正式模型会在 50×50 棋盘上继续运行到碰撞或填满棋盘，长度 300 不是终止条件。

## 最终网络

```text
4 个候选方向 × 9 个实时传感输入
        ↓（四个方向共享参数）
128 个神经元
        ↓
8 层深层 ReLU 网络（每层 128）
        ↓
4 个原始 logits
        ↓
argmax → env.step
```

推理时没有动作 mask、空间安全盾、合法动作替换或 fallback。V3 的八个隐藏层均为真正的稠密矩阵，每层 `16,384/16,384` 个权重非零，不再使用单位矩阵传递。GUI 上半区逐个显示四个候选方向的 4,608 个实时神经元（不抽样），下半区以一像素一个连接的方式显示全部 132,352 个权重。青色表示正权重、粉色表示负权重，明暗表示绝对值大小；鼠标悬停可查看精确数值。实时神经元使用 36 张批量图像刷新，避免逐个修改数千个 Tk Canvas 对象造成崩溃。

## 最终验收

- 训练随机种子：`20261101`
- 检查点哈希派生验收种子：`139612045`
- 100/100 局达到长度 `321`
- 动作改写次数：`0`
- 食物反事实改变 logits：`99.8%`
- 食物反事实改变 argmax：`95.2%`
- 多后继格比例：`97.08%`
- 相对固定逐行扫描的中位步数比例：`10.27%`

完整报告：`models/evaluation_v3_321.json`。

## 安装

需要 Python 3.11 或更高版本。CPU 可以直接运行；如已安装兼容 CUDA 的 PyTorch，训练和批量评估会自动使用对应环境。

```powershell
git clone https://github.com/pgd-LC2/snake.git
cd snake
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行 GUI

```powershell
python -u gui.py --target 321 --speed 20
```

## 重新训练与测试

```powershell
python -u train.py --seed 20261101 --target 321
python -u evaluate.py --model models\snake_policy.pt --episodes 100 --target 321
python -m unittest -v test_food_policy.py test_hamiltonian.py test_snake.py
```

## 核心文件

- `models/snake_policy.pt`：正式深层网络模型
- `models/snake_policy.json`：训练报告
- `models/evaluation_v3_321.json`：100 局验收汇总
- `models/evaluation_v3_321_episodes.csv`：逐局数据
- `gui.py`：新版实时网络 GUI
- `train.py`：可复现训练与深层模型导出
- `evaluate.py`：反刷分验收
- `snake_core.py`：游戏规则和输入编码
