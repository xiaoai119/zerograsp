# PickSingle 常驻执行 Worker

## 目的

`run_zerograsp_depth_ab_batch.py` 默认会为每个抓取启动一个新的 Python 进程，并重新创建：

- ManiSkill 环境；
- RGB-D 相机；
- cuRobo planner；
- CUDA graph 和规划 warmup。

这些初始化远慢于真正的 cuRobo 求解。为了加速使用已有 ZeroGrasp 候选的 PickSingle 批量实验，批处理脚本增加了：

```text
--persistent-worker
```

## 使用方法

```bash
PYTHONPATH=. python \
  -m maniskill_curobo.scripts.run_zerograsp_depth_ab_batch \
  --env-id PickSingleYCB-v1 \
  --seed-start 1 \
  --seed-end 200 \
  --output-root maniskill_curobo/runs/example_persistent_batch \
  --baseline-depth-scale 0.0 \
  --depth-scale 1.0 \
  --settle-before-export-steps 20 \
  --persistent-worker
```

既可以现场重新拍摄并生成候选，也可以通过
`--reuse-candidate-root` 使用已有候选。已有候选目录需要保持以下结构：

```text
candidate_root/
  seed001_base/
    zg_output/
      recommended_grasp_top1.json
  seed002_base/
    zg_output/
      recommended_grasp_top1.json
```

父进程会自动使用 `--maniskill-python` 指定的解释器启动一个常驻子进程。

## 复用内容

- ManiSkill 环境只创建一次；
- 相机和渲染器只初始化一次；
- 现场生成候选时，RGB-D 输入导出环境也只创建一次；
- cuRobo planner 只创建并 warmup 一次；
- 每个 variant 仍独立调用 `env.reset(seed=...)`；
- 每个 variant 仍保存独立的日志、manifest、轨迹和视频；
- 每次复用 planner 前会重置随机种子；
- 原有单 episode 执行逻辑和成功判定保持不变。

## 安全限制

当前常驻 worker 只支持：

```text
PickSingleYCB-v1
scene_source = maniskill
```

worker 会比较每个 episode 导出的 cuRobo 场景。只要碰撞世界与第一次不同，就会拒绝继续复用 planner。

因此当前不支持 PickClutter。PickClutter 的障碍物随 seed 改变，需要先实现 cuRobo 动态 world update，不能直接复用第一帧场景。

## 实测结果

测试设置：

- PickSingle seed 1-2；
- 每个 seed 执行 zero 和 full depth；
- 共 4 个 episode；
- 关闭视频；
- `close_steps=20`；
- `settle_steps=50`。

结果：

| 模式 | 四个 episode 总时间 | 单 episode 情况 |
|---|---:|---|
| 原逐进程模式 | 80.49 秒 | 约 20.1 秒 |
| 常驻 worker | 35.12 秒 | 首次 18.44 秒，后续约 4.8-5.2 秒 |

小样本总耗时降低约 56%，吞吐量提高约 2.29 倍。批次越大，首次初始化成本占比越低，整体收益越接近后续 episode 的速度。

两种模式的以下结果逐项一致：

- success/object_not_lifted；
- lift 成功标记；
- 最终 lift 高度；
- zero 与 full depth 的改善关系。

## 视频内存优化

原录像器会把所有 `1280x1024` RGB 帧保存在内存，结束时再一次性堆叠并编码。常驻执行连续录制时，这会造成较高内存峰值。

录像器现已改为逐帧写入 MP4：

- 两个连续录像 episode 的峰值内存从约 5.1 GB 降到约 2.6 GB；
- 视频时长和内容保持正常；
- 运行时间基本不变。

## 输出中的复用信息

每个 variant 的 summary command 字段会增加：

```json
{
  "persistent_worker": true,
  "episode_index": 2,
  "environment_reused": true,
  "planner_reused": true
}
```

首个 episode 的两个 `reused` 字段为 `false`，后续 episode 为 `true`。

## ZeroGrasp 常驻推理 Worker

当批处理没有使用 `--reuse-candidate-root`、需要现场生成 ZeroGrasp
候选时，脚本会默认启用另一套独立 worker：

```text
--persistent-zerograsp-worker
```

它运行在 `--zerograsp-python` 指定的环境中，只加载一次 ZeroGrasp
checkpoint 和 CUDA 上下文，随后通过 JSON-lines 请求依次处理每个 seed。
这不会改变每个 seed 的输入、碰撞检测、随机种子或输出目录。

如需退回原来每个 seed 单独启动一次 ZeroGrasp 的方式：

```text
--no-persistent-zerograsp-worker
```

RTX 3090、开启 ZeroGrasp 碰撞检测的实测结果：

| 项目 | 耗时 |
|---|---:|
| worker 启动、依赖导入和模型加载 | 10.67 秒，仅一次 |
| worker 第一次请求 | 6.03 秒 |
| worker 后续稳定请求 | 4.08 秒/seed |
| 原独立 CLI 冷启动 | 18.66 秒/seed |

两个 seed 的总墙钟时间从约 37.32 秒降到 22.47 秒，降低约 40%。
按 200 个 seed 估算，ZeroGrasp 阶段约从 62 分钟降到 14 分钟。
这里不包含 ManiSkill 输入导出和 cuRobo 执行时间。

worker 输出保留原有结构：

```text
seedNNN/setup/zg_output/
  raw_outputs/
  recommended_grasp_top1.json
  run_report.json
```

公共 worker 日志位于：

```text
output_root/worker_logs/
  zerograsp_worker.stdout.log
  zerograsp_worker.stderr.log
```
