# ManiSkill Codex ZeroGrasp 抓取执行设计

## 背景

`maniskill_codex/` 目前是空目录。目标是在该目录下新增独立实现：读取 ZeroGrasp 已输出的抓取结果，并在 ManiSkill 仿真环境中用 Panda 执行抓取。

现有 `maniskills/` 目录可以作为场景参考，但新实现不复用其中代码结构和函数实现。第一版聚焦抓取执行，不做 ZeroGrasp 输出可视化。

## 范围

本次实现包含：

- 创建 `PickSingleYCB-v1` + `panda` 的 ManiSkill 环境。
- 读取 ZeroGrasp 离线输出：
  - 优先读取 `recommended_grasp_top1.json`。
  - 若没有该文件，则从 `raw_outputs/*.grasp.npy` 中选最高分抓取。
- 将 ZeroGrasp 相机坐标系下的抓取中心转换到机器人 base 坐标系。
- 执行基础抓取序列：pre-grasp、descend、close、lift。
- 打印关键调试信息，包括抓取分数、目标点、每阶段实际末端位置和 success 信息。
- 为输出解析和坐标转换写轻量单元测试。

本次不包含：

- 不实时调用 ZeroGrasp HTTP 服务。
- 不在同一进程中加载 ZeroGrasp 模型。
- 不实现抓取位姿可视化 marker。
- 不重写 ManiSkill 内置任务。

## 架构

新增 `maniskill_codex/` Python 包，建议文件如下：

- `maniskill_codex/zerograsp_outputs.py`
  负责读取 `recommended_grasp_top1.json` 或 `*.grasp.npy`，返回统一的抓取记录。
- `maniskill_codex/transforms.py`
  负责 OpenCV camera frame、SAPIEN camera frame、world frame、robot base frame 之间的转换。
- `maniskill_codex/execute_zerograsp_pick.py`
  命令行入口，创建 ManiSkill 环境、加载抓取结果、执行动作序列。
- `maniskill_codex/tests/`
  放解析和转换相关测试。

## 数据流

1. 用户运行脚本，传入 ZeroGrasp 输出目录，例如 `--zerograsp-output output`。
2. 脚本查找抓取结果：
   - `output/recommended_grasp_top1.json`
   - 或 `output/raw_outputs/*.grasp.npy`
3. 抓取记录统一为：
   - `score`
   - `width_m`
   - `height_m`
   - `depth_m`
   - `rotation_matrix_camera`
   - `translation_m_camera`
   - `source`
4. 创建 ManiSkill 环境并 reset。
5. 从环境相机读取 model matrix，从机器人读取 base pose。
6. 将 `translation_m_camera` 转换到 base 坐标系。
7. 生成动作阶段：
   - `pre`: 抓取点上方或沿接近方向退后一段，夹爪张开。
   - `descend`: 移到抓取点，夹爪张开。
   - `close`: 原地闭合夹爪。
   - `lift`: 抬升目标，夹爪保持闭合。
8. 每阶段 step 多次，结束后打印 success 或 truncated。

## 坐标约定

ZeroGrasp 输出使用 OpenCV 相机坐标：

- `+X`: 图像右侧
- `+Y`: 图像下方
- `+Z`: 相机前方

SAPIEN/ManiSkill 相机 point map 常用约定可通过符号矩阵转换：

```text
S = diag([1, -1, -1])
p_sapien_camera = S @ p_opencv_camera
```

然后使用 ManiSkill 相机的 model matrix 转世界坐标：

```text
p_world = camera_model_matrix @ [p_sapien_camera, 1]
```

最后用机器人 base pose 的逆矩阵转到 base 坐标：

```text
p_base = inv(T_world_base) @ [p_world, 1]
```

第一版只用 ZeroGrasp 的抓取中心作为控制目标。抓取朝向会被解析并保留在数据结构中，但执行阶段先采用固定末端姿态，以降低控制复杂度。

## 错误处理

- 找不到抓取输出时，报清楚应传入的目录结构。
- 抓取数组为空时跳过该文件，若所有文件为空则报错。
- JSON 字段缺失时给出字段名和来源文件。
- ManiSkill 环境不可导入时提示需要 `conda activate maniskill`。
- 执行阶段若 episode truncated，打印当前阶段和最后 info。

## 测试

测试重点放在不依赖 GPU 和仿真的逻辑：

- `recommended_grasp_top1.json` 能被解析为统一抓取记录。
- 多个 `*.grasp.npy` 能按 score 选出最高分。
- OpenCV camera 到 base 的矩阵转换在人工构造矩阵下结果正确。
- 找不到输出、空数组、缺字段时错误信息可读。

完整仿真 smoke test 使用：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maniskill
PYTHONPATH=. python -m maniskill_codex.execute_zerograsp_pick --zerograsp-output output --episodes 1 --seed 42
```

## 验收标准

- 新代码只写入 `maniskill_codex/` 和对应测试/文档，不修改 `maniskills/`。
- 脚本能从现有 `output/raw_outputs/*.grasp.npy` 读到最高分抓取。
- 单元测试通过。
- 在 `maniskill` conda 环境中，脚本能启动 `PickSingleYCB-v1` 并执行完整 pre/descend/close/lift 流程。

