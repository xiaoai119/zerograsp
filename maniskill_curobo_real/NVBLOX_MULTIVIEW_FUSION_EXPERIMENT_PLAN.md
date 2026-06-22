# 多视角点云融合 / nvblox 碰撞场景实验计划

## 结论先说

这个方向可行，而且很适合验证“单视角 ZeroGrasp 重建点云是否导致碰撞世界不完整”这个问题。

不再把“直接拼接三视角 ZeroGrasp 点云”作为主线。这个方法看起来最容易实现，但问题也最明显：每个视角的深度误差、外参误差、实例边界误差都会叠加，最后点云容易变厚、变毛、数量暴涨，cuRobo 会把它当成更大的障碍，反而更保守。

更合理的主线是：

1. 保留当前单视角 M4C-ZG 作为 baseline。
2. 三个视角采 RGB-D 和准确相机位姿。
3. 用 ZeroGrasp 的感知结果做目标排除、mask guidance、抓取候选来源。
4. 用 nvblox 直接融合 RGB-D，生成 TSDF/ESDF。
5. 让 cuRobo 基于 nvblox ESDF 做碰撞查询或把 nvblox 结果导出为当前 M4C 可读的 voxel ESDF。

简单说：不是“拼点云”，而是“多视角 RGB-D 融合成连续场景距离场”。

## nvblox 适不适合

适合。nvblox 本身就是给 RGB-D 相机做实时 3D 重建和 ESDF 查询的库，输入是多帧 depth/color + camera pose，输出是 TSDF/ESDF/mesh。cuRobo 官方也提供了 nvblox 接口，可以把 depth 图融合进 nvblox ESDF，再让 cuRobo 做碰撞查询。

它和我们现在 M4C 的关系可以这样理解：

- 现在 M4C：我们自己把点云体素化，算一个 ESDF `.npz`。
- nvblox：由 nvblox 管理 TSDF/ESDF 地图，然后 cuRobo 直接查询这个地图。

两者本质都是给 cuRobo 一个“空间里哪里有障碍、离障碍多远”的场景模型，只是地图构建方式不同。

## 当前仓库已有基础

当前 `maniskill_curobo_real` 里已经有这些基础能力：

- `scene_builder.py`
  - `build_zerograsp_instance_voxel_esdf_scene`
  - 把 ZeroGrasp reconstructed instances 转成 M4C-ZG voxel ESDF。
- `zerograsp_reconstruction.py`
  - 读取 ZeroGrasp 的 `*.reconstruction.npz`。
  - 把相机坐标系点云转换到 robot base 坐标系。
- `run_world_collision_stages.py`
  - 已支持 `m4c`。
  - 已支持 `--point-cloud-source zerograsp_reconstruction`。

所以现有代码仍然有价值，但它主要作为 baseline 和数据接口复用，不再作为正式多视角融合主线。正式主线应该新增 nvblox RGB-D fusion，而不是把多个 ZeroGrasp reconstruction point cloud 直接相加。

## 实验目标

验证三件事：

1. 三视角 RGB-D nvblox 融合，是否比单视角 M4C-ZG 更接近 oracle / ManiSkill 真值碰撞场景。
2. nvblox 融合后的碰撞世界，是否提升 cuRobo 规划成功率和抓取 lift 成功率。
3. nvblox 是否比当前 numpy/scipy voxel ESDF 更稳定、更快或更接近真实上机链路。

## 三个视角怎么采

每个 seed 固定在场景 settle 后采三张 RGB-D：

1. 中间视角：当前机器人头部 / ZeroGrasp 主视角。
2. 左侧斜视角：绕桌面目标区域 yaw 左偏，例如 25-35 度。
3. 右侧斜视角：绕桌面目标区域 yaw 右偏，例如 25-35 度。

其中一个视角必须直接复用当前已有的 RGB-D 相机，用来保持和现有单视角链路可比；另外两个补充视角要尽量靠近桌面和目标区域，不能为了仿真看得全而把相机放得太远。真实机器人上远距离 RGB-D 的深度误差、标定误差和遮挡都会明显放大，所以补充视角应该模拟真机可安装或可运动到的近距离观察位置。

三张图都要保存：

- RGB
- depth
- camera intrinsics
- camera pose，即 `T_world_camera` / `T_base_camera`
- ZeroGrasp 输入 bundle
- ZeroGrasp 输出 grasp / reconstruction / masks

重点是每个视角的外参必须准确，否则融合出来的点云会“重影”或变厚，反而让 cuRobo 过度保守。

## ZeroGrasp 在这里怎么用

ZeroGrasp 不直接等于 nvblox。

比较合理的分工是：

- ZeroGrasp 负责：
  - 从每个视角输出抓取候选。
  - 输出 mask / target guidance / reconstruction diagnostic。
  - 提供目标物体识别或目标区域 mask，帮助从碰撞世界里排除目标物体。
- 融合器负责：
  - 把三个视角的 RGB-D 按相机位姿融合进 nvblox。
  - 在融合前或融合后应用目标排除 / 非目标过滤。
  - 生成 nvblox TSDF/ESDF，供 cuRobo 查询。

注意：如果要用真正 nvblox，最自然的输入是 RGB-D + pose，而不是把 ZeroGrasp 点云硬塞给 nvblox。ZeroGrasp 更适合作为 mask / target exclusion / instance guidance。

## 正式比较对象

### Baseline：当前单视角 M4C-ZG

名称：`M4C-ZG-single-view`

做法：

- 只用当前主相机视角。
- 使用当前 `build_zerograsp_instance_voxel_esdf_scene`。
- 这是我们要比较的 baseline。

### Proposed：三视角 RGB-D nvblox 融合

名称：`M5-nvblox-rgbd-fusion`

做法：

- 三个视角采 RGB-D + pose。
- 用 ZeroGrasp / mask 过滤目标物体或给层级分割提供 guidance。
- 把 depth + pose 送入 nvblox。
- 得到 nvblox TSDF/ESDF。
- 两条接入路线：
  - 路线 A：走 cuRobo `WorldBloxCollision`，直接查 nvblox ESDF。
  - 路线 B：从 nvblox 导出 voxel/mesh，再转换成当前 `.npz` 形式。

优点：

- 更接近真机方案。
- 官方就是为 RGB-D 多帧融合和 ESDF 查询做的。

缺点：

- 当前环境没有安装 nvblox。
- Python/CUDA/ABI 依赖更麻烦。
- cuRobo 官方也提醒，相机感知碰撞规划在密集遮挡环境里仍然容易失败。

### Reference：oracle / ManiSkill 真值碰撞场景

名称：`oracle-M4C` 或现有 ManiSkill 真值碰撞场景。

用途：

- 不作为真机可实现方案。
- 只用来衡量 reconstruction / ESDF 的几何上限。
- 判断 nvblox 是更接近真值，还是只是让场景变得更保守。

## 非正式诊断工具

### 三视角 ZeroGrasp 点云拼接诊断

名称：`M4C-ZG-multiview-pointcloud-debug`

这个不作为正式实验版本，也不放进最终性能表，只用于检查：

- 三个视角的相机外参有没有对齐。
- ZeroGrasp reconstruction 在不同视角下是否有明显重影。
- target exclusion 是否把目标物体错误放进障碍物。
- nvblox 融合异常时，问题来自 RGB-D 外参、mask，还是来自 nvblox 参数。

它的意义不是提升性能，而是排查问题。正式性能比较只看：

```text
M4C-ZG-single-view  vs  M5-nvblox-rgbd-fusion  vs  oracle-M4C
```

## 重建精度怎么评估

只看抓取成功率不够，因为抓取还受 ZeroGrasp 候选、depth scale、IK、轨迹规划、控制误差影响。这里要单独评估碰撞场景质量。

建议指标：

1. voxel occupancy IoU
   - 把 M4C-ZG 单视角、nvblox 三视角融合、oracle M4C 全部转成同一个 voxel grid。
   - 看占据体素和 oracle 的交并比。

2. precision / recall / F1
   - precision 低：说明假障碍太多，会让 cuRobo 过度保守。
   - recall 低：说明漏障碍，会让机械臂撞物体或桌子。

3. surface / voxel distance
   - nvblox surface / occupied voxels 和 oracle instance surface 的几何距离。

4. target leakage
   - 目标物体是否被错误放进碰撞世界。
   - 这个很关键，因为目标如果被当障碍，grasp 阶段容易 planning failed。

5. cuRobo 结果
   - pre-grasp planning success
   - grasp planning success
   - lift success
   - planning failed / object_not_lifted / collision-like failure 分类。

## 推荐执行顺序

### Phase 1：采集三视角数据

在 `maniskill_curobo_real` 中新增三视角采集脚本：

- 输入：task、seed range、camera preset。
- 输出：每个 seed 一个目录，包含 `view_0/1/2` 的 RGB-D、相机内外参、ZeroGrasp 输入图。

先跑少量 seed，例如 `1-20`，确认三视角图像和外参没问题。

当前已实现：

- `capture_multiview_rgbd.py`
  - 每个 seed 导出三个 view。
  - `view_0_existing_rgbd` 复用当前已有 RGB-D 相机。
  - `view_1_close_left` / `view_2_close_right` 是近距离左右补充视角。
  - 每个 view 保存 `rgb.png`、`depth.png`、`mask.png`、`rgbd.npz`、`camera.json`、`view_metadata.json`。
  - `view_metadata.json` 里保存 `world_from_camera`、`world_from_base`、`base_from_camera`，后续可直接给 nvblox / cuRobo 使用。
- 已用 seed1 跑过低分辨率 smoke test，三视角数据可以正常落盘。

### Phase 2：可选三视角点云诊断，不做正式路线

新增轻量诊断模块：

- 读取三个 view 的 ZeroGrasp reconstruction。
- 全部变换到 base frame。
- 合并非目标点云后只写出可视化图。
- 检查三视角是否对齐、是否有明显重影、目标是否泄漏到障碍物。
- 不把这个结果作为正式 cuRobo 性能路线。

### Phase 3：接 nvblox RGB-D 融合

实现真正的 RGB-D 融合：

- 安装/编译 nvblox 或 nvblox_torch。
- 三个视角的 depth + pose 依次 integrate 到 nvblox。
- 用 ZeroGrasp / mask 过滤目标物体或做 target exclusion。
- 得到 TSDF/ESDF。
- 优先尝试导出为当前 M4C 可读的 voxel ESDF，后续再接 cuRobo `WorldBloxCollision`。

当前已实现：

- `nvblox_fusion.py`
  - 检测 `nvblox_torch` / `nvblox` / `pynvblox` 是否可用。
  - 检查三视角采集目录是否完整。
  - 当前环境尚未安装外部 nvblox 后端，所以 `external-nvblox` 会明确报错，而不是静默生成错误结果。
  - 已新增 `curobo-mapper` 后端，使用 cuRobo 自带 block-sparse TSDF/ESDF mapper 融合三视角 RGB-D，并导出当前 M4C 可读的 `curobo_scene_voxel.npz`。
  - 已用 seed1 低分辨率 smoke test 跑通：三视角采集 -> cuRobo mapper 融合 -> cuRobo 读取 voxel `.npz` 执行。当前 seed1 结果为 candidate selection planning failed，但这已经进入真实规划阶段，不再是文件格式/后端接口问题。

新增脚本：

- `run_multiview_nvblox_benchmark.py`
  - 对每个 seed 自动执行：采三视角 RGB-D -> cuRobo mapper 融合 ESDF -> 复用现有 ZeroGrasp candidate 执行抓取。
  - 默认复用 `maniskill_curobo/runs/depth_corrected_settle20_seed1_200_rerun` 中已有候选，避免把 ZeroGrasp 重新推理耗时混进碰撞世界实验。

### Phase 4：重建精度评估

对同一批 seed 生成：

- single-view M4C-ZG
- multiview nvblox
- oracle M4C

输出 CSV / Markdown：

- occupied voxels
- IoU
- precision
- recall
- F1
- target leakage
- table / object 点数统计

### Phase 5：抓取链路评测

在 seed `1-200` 上对比：

- M4C-ZG-single-view
- M5-nvblox-rgbd-fusion
- oracle M4C

确认 nvblox 多视角融合是否真正提升规划和抓取。

## 风险和注意点

1. 多视角不是自动补全所有几何。
   - 三个视角都没看到的背面，仍然不能凭空恢复。

2. 外参误差会让障碍物变厚。
   - 仿真里外参完美，真机上必须标定好 `T_base_camera`。

3. 目标物体排除很关键。
   - 如果目标被放进碰撞世界，grasp 阶段规划会更容易失败。

4. 多视角可能提升 recall，但降低 precision。
   - 看见更多障碍是好事，但噪声也会更多。
   - 所以不能用粗暴点云拼接作为主线，要让 nvblox 用 TSDF/ESDF 方式融合。

5. nvblox 不是万能。
   - 它适合 sparse obstacle 和局部地图，密集遮挡桌面操作仍然需要目标分割、滤波、地图清理策略。

## 预期结果

如果 nvblox 多视角融合有效，应该看到：

- M5-nvblox 的 obstacle recall 高于 single-view。
- 背侧/侧面缺失减少。
- grasp planning failed 下降。
- 机械臂运动中撞桌子/撞障碍的情况减少。
- 但可能出现更保守的规划，需要调 voxel size、dilation、target exclusion。

如果多视角融合无效，常见原因会是：

- depth / pose 融合后假障碍太多。
- 目标排除不准。
- nvblox 参数过保守，例如 voxel size、truncation distance、decay/clearing 策略不合适。
- 桌面/目标/非目标边界没分干净。

## 最小可跑版本

正式最小版本需要 nvblox：

```text
三视角 RGB-D
  -> ZeroGrasp mask / target guidance
  -> nvblox integrate depth + pose
  -> nvblox TSDF/ESDF
  -> 导出/接入 cuRobo collision world
  -> cuRobo seed1-200 评测
```

点云拼接版只保留为调试工具，用来检查三视角外参和 mask 是否大致合理，不作为最终实验结论来源。

## 2026-06-18 初次 M5 小批量运行记录

命令：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python -m maniskill_curobo_real.run_multiview_nvblox_benchmark \
  --seed-start 1 --seed-end 20 \
  --output-root maniskill_curobo_real/runs/m5_multiview_curobo_mapper_seed1_20
```

说明：

- 当前环境没有安装官方 NVIDIA nvblox Python 接口。
- 本轮使用 cuRobo 自带 mapper 做 TSDF/ESDF 融合，作为 nvblox-style 最小闭环验证。
- 三个视角包含一个现有 RGB-D 视角和两个近桌面辅助视角。

结果：

- 总计 20 个 seed。
- lift success: 0 / 20。
- `planning_failed_candidate_selection`: 17 / 20。
- `planning_failed_grasp`: 1 / 20。
- `object_not_lifted`: 2 / 20。
- 平均总耗时约 24.0 秒 / seed。
- 平均 fusion 耗时约 1.33 秒 / seed。

初步判断：

- 这轮不能直接说明“多视角融合无效”。
- 主要问题是当前 cuRobo mapper 生成的 ESDF/voxel 表达偏异常：20 个 seed 里有 17 个 seed 的 `surface_band_voxels = 0`，也就是融合出来的障碍几何没有有效落到可用于规划的表面带里。
- 下一步应该先做 M5 地图可视化与 M4C-ZG / oracle M4C 对齐检查，再调 `voxel_size`、`truncation_distance`、深度坐标转换、target exclusion 和 ESDF 导出格式，而不是直接扩大 seed 数。

## 2026-06-18 M5 可视化检查记录

新增脚本：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python -m maniskill_curobo_real.visualize_m5_multiview_fusion \
  --root maniskill_curobo_real/runs/m5_multiview_curobo_mapper_seed1_20 \
  --seeds $(seq 1 20)
```

输出目录：

```text
maniskill_curobo_real/runs/m5_multiview_curobo_mapper_seed1_20/visualization/
```

每个 seed 的图包含：

- 三个 RGB-D 视角原图。
- 三视角反投影到 robot base 后的输入点云。
- cuRobo mapper 输出的 ESDF / near-surface voxels。
- top-down ESDF heatmap。

关键发现：

- 三视角 RGB-D 输入本身基本正常，桌面和目标物体能在 robot base 中反投影到合理位置。
- 20 个 seed 中有 17 个 seed 的 ESDF 全部是 `10000`，也就是 mapper 没有写出有效距离场。
- 非空的 seed，例如 seed6 / seed17，ESDF voxels 主要集中在画面上缘的机器人自身/夹爪区域，而不是桌面物体区域。
- 当前只排除了 target object mask，没有排除机器人自身可见部分；这些机器人自身深度被误融合进 obstacle world，会导致 cuRobo 规划异常保守或候选筛选失败。

下一步修正方向：

1. 给 M5 输入增加 robot-self mask / robot-depth exclusion。
   - 仿真里可以先用 segmentation 排除 panda / robot actor。
   - 真机上对应做法是用已知机器人模型和当前关节状态渲染/投影 robot mask，或用近似工作区裁剪去掉相机上缘的机器人自遮挡。

2. 增加 workspace crop。
   - 只把桌面上方一定高度范围内、靠近操作区的点送入 mapper。
   - 避免背景地面、机器人本体、远处点进入 ESDF。

3. 再检查 cuRobo mapper 的 integration / compute_esdf 参数。
   - 现在很多 seed 输入点云正常但 ESDF 仍为空，说明 mapper 对当前相机位姿、深度单位、ESDF origin / grid center 仍可能有不匹配。

4. 完成上述修正后再重新跑 seed1-20。
   - 如果 seed1-20 的 ESDF 至少稳定覆盖非目标物体和桌面附近障碍，再扩大到 seed1-200。

## 2026-06-18 PickClutter M5 小批量运行记录

用户指出：多视角融合要验证的是“非目标物体/桌面障碍是否能被更准确地放入 cuRobo collision world”，因此 PickClutter 比 PickSingle 更合适。PickSingle 中目标物体被排除后，场景里几乎没有需要融合的非目标障碍，容易得到误导性的空 ESDF 结果。

本轮改用 PickClutterYCB-v1，复用已生成的 full-depth ZeroGrasp 候选，只替换碰撞世界为三视角 RGB-D 融合后的 cuRobo mapper ESDF。

命令：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python -m maniskill_curobo_real.run_multiview_nvblox_benchmark \
  --env-id PickClutterYCB-v1 \
  --seed-start 2 --seed-end 21 \
  --reuse-candidate-root maniskill_curobo_real/runs/pickclutter_full_depth_m0_seed1_200_candidate_reuse \
  --output-root maniskill_curobo_real/runs/m5_multiview_curobo_mapper_pickclutter_seed2_21
```

结果：

- 总计 20 个 seed。
- lift success: 9 / 20。
- 成功 seed: 3, 5, 8, 10, 11, 12, 15, 16, 21。
- `object_not_lifted`: 4 个，seed 2, 7, 13, 18。
- `planning_failed_pre`: 1 个，seed 4。
- `planning_failed_grasp`: 1 个，seed 20。
- `planning_failed_lift`: 2 个，seed 6, 19。
- `planning_failed_candidate_selection`: 3 个，seed 9, 14, 17。
- 平均总耗时约 22.46 秒 / seed。

可视化命令：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python -m maniskill_curobo_real.visualize_m5_multiview_fusion \
  --root maniskill_curobo_real/runs/m5_multiview_curobo_mapper_pickclutter_seed2_21 \
  --seeds $(seq 2 21)
```

可视化输出目录：

```text
maniskill_curobo_real/runs/m5_multiview_curobo_mapper_pickclutter_seed2_21/visualization/
```

可视化检查结果：

- 20 / 20 个 seed 都有有效 ESDF，不再出现 PickSingle 中大面积 `all_default_esdf = true` 的问题。
- occupied voxels: min / avg / max = 2757 / 3900.85 / 5237。
- surface band voxels: min / avg / max = 5701 / 8026.3 / 10621。
- 三个 RGB-D 视角能看到桌面和多个非目标物体。
- 输入点云反投影到 robot base 后位置合理。
- ESDF near-surface voxels 覆盖了桌面前方可见障碍区域，说明 PickClutter 才是当前 M5 多视角融合更合理的验证场景。

当前结论：

- 之前 PickSingle 的 M5 结果不能代表多视角融合质量，因为目标物体被排除后几乎没有非目标障碍可建图。
- PickClutter 小批量结果说明三视角融合链路可以生成有效 cuRobo mapper ESDF，并且能跑通抓取闭环。
- 下一步应在 PickClutter 上做严格对照：单视角 M4C-ZG vs 三视角 M5，统一候选来源、top-k、full-depth 设置，再看 planning failure 和 lift success 是否改善。

## 2026-06-18 M5 方案修正：从整体融合改为实例级重建

当前 M5 小批量使用的是“扣掉目标物体后，把剩余所有 depth 整体融合成一个 ESDF”的方式。这条链路可以作为 baseline，但它过于粗糙：

- 它不知道 clutter 中每个物体是谁。
- 多个物体、桌面边缘、机器人可见部分、噪声点会被混成一个障碍场。
- 一旦某个视角有错误 depth，错误会直接进入整体 ESDF。
- 后续很难判断失败到底来自哪个物体、哪个 mask、哪个视角。

更合理的主线应改成 instance-aware M5：

```text
三视角 RGB-D
  -> 每个视角做实例分割
  -> 找出任务目标 instance，并排除目标
  -> 对非目标桌面物体做跨视角实例关联
  -> 每个物体单独融合 / 重建点云
  -> 每个物体生成 OBB / mesh / voxel / ESDF 碰撞体
  -> 合成 cuRobo collision world
```

### 仿真最小实现

在 ManiSkill 中先使用 oracle instance segmentation 验证逻辑：

- 从每个视角保存 `PositionSegmentation` / instance id。
- `task-target` 对应的 instance id 作为目标物体，直接从 collision world 排除。
- 对其他 instance id 分别反投影出 3D 点云。
- 跨视角用 segmentation id 直接对齐，同一个 id 的点云合并。
- 每个非目标 instance 独立生成碰撞体。
- 暂时不单独构建桌子碰撞体，先只验证桌面物体的实例级重建质量。

这一步不代表真机最终输入，但它能验证“实例级多视角融合”本身是否比整体 ESDF 更稳定。

### 真机对应实现

真机上没有 ManiSkill instance id，因此需要换成真实感知：

- 目标实例：由任务指令、ZeroGrasp/GraspNet 输入 mask、目标检测或 Grounded-SAM 给出。
- 非目标桌面物体：由 SAM / Mask2Former / Grounded-SAM / CNOS 等实例分割得到。
- 跨视角关联：用 3D centroid、mask 投影 IoU、颜色/几何特征做匹配。
- 桌子：当前阶段先不加入 collision world；后续如果需要，再作为单独 ablation 用平面拟合或固定桌面模型加入。

### 需要比较的版本

后续 PickClutter 上至少比较四个版本：

1. `M4C-ZG`：单视角 ZeroGrasp 重建点云生成 voxel ESDF。
2. `M5-global`：当前整体 depth 融合 baseline。
3. `M5-instance-obb-no-table`：三视角实例点云融合，每个非目标物体生成 OBB，不构建桌子。
4. `M5-instance-esdf-no-table`：三视角实例点云融合，每个非目标物体生成 voxel/ESDF，不构建桌子。

### 预期收益

- 更少误把目标物体放进碰撞世界。
- 更少把多个物体粘成一大团假障碍。
- 对每个失败案例可以定位到具体 instance。
- 更接近真机部署时“实例分割 + 局部物体建模”的路线。

### 关键风险

- 真机实例分割不一定稳定，尤其是透明、反光、遮挡、接触桌面的物体。
- 跨视角实例关联如果错了，会把两个物体合并或把一个物体拆碎。
- 单物体重建仍然只有可见表面，背面需要用补全、padding、OBB 或保守 voxel 膨胀处理。
- 当前 no-table 版本可能放过机械臂/夹爪与桌面的碰撞风险；这是有意为之，用来先隔离验证物体实例建模质量。桌子碰撞应在后续单独实验中再加入。

## 2026-06-18 M5 Instance No-Table 实现与小批量结果

新增实现：

- `maniskill_curobo_real/multiview_instance_scene.py`
  - 读取三视角 RGB-D bundle。
  - 根据每个视角的 instance mask 和 `segmentation_id` 合并同一物体的点云。
  - 排除 `is_task_target = true` 的目标物体。
  - 不构建桌子。
  - 支持两种碰撞表达：
    - `m5_multiview_instance_obb_no_table`
    - `m5_multiview_instance_voxel_esdf_no_table`

- `maniskill_curobo_real/run_multiview_instance_benchmark.py`
  - 复用已有 ZeroGrasp full-depth top-k 候选。
  - 复用三视角 RGB-D 输入。
  - 复用 persistent cuRobo 执行链路。

- `maniskill_curobo_real/run_world_collision_stages.py`
  - `.npz` voxel scene loader 支持 no-table 场景。
  - 如果 npz 中有 `table_pose/table_dims` 就加载桌子；如果没有，就只加载 voxel grid。

运行命令：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python -m maniskill_curobo_real.run_multiview_instance_benchmark \
  --env-id PickClutterYCB-v1 \
  --seed-start 2 --seed-end 21 \
  --stage m5-instance-obb-no-table \
  --reuse-existing \
  --reuse-candidate-root maniskill_curobo_real/runs/pickclutter_full_depth_m0_seed1_200_candidate_reuse \
  --output-root maniskill_curobo_real/runs/m5_multiview_curobo_mapper_pickclutter_seed2_21
```

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python -m maniskill_curobo_real.run_multiview_instance_benchmark \
  --env-id PickClutterYCB-v1 \
  --seed-start 2 --seed-end 21 \
  --stage m5-instance-esdf-no-table \
  --reuse-existing \
  --reuse-candidate-root maniskill_curobo_real/runs/pickclutter_full_depth_m0_seed1_200_candidate_reuse \
  --output-root maniskill_curobo_real/runs/m5_multiview_curobo_mapper_pickclutter_seed2_21
```

同一批 PickClutter seed2-21 结果：

| 版本 | 桌子 | lift success | 成功 seed | 主要失败 |
| --- | --- | ---: | --- | --- |
| M5-global ESDF baseline | 有默认 table + 整体 ESDF | 9 / 20 | 3, 5, 8, 10, 11, 12, 15, 16, 21 | object_not_lifted 4, candidate_selection 3, pre/lift/grasp planning 4 |
| M5-instance-OBB-no-table | 无 | 4 / 20 | 5, 16, 20, 21 | object_not_lifted 10, candidate_selection 5, grasp planning 1 |
| M5-instance-ESDF-no-table | 无 | 7 / 20 | 5, 9, 10, 12, 16, 20, 21 | object_not_lifted 12, candidate_selection 1 |

实例重建统计：

- 每个 seed 看到的 instance 数约 5-7 个。
- 实际进入 collision world 的非目标 instance 数：
  - min / avg / max = 3 / 5.2 / 7。
- `table_included = false`，确认 no-table 设置生效。
- instance ESDF occupied voxels：
  - min / avg / max = 255 / 1082.6 / 2516。

初步结论：

- instance-aware no-table 链路已经跑通。
- `M5-instance-ESDF-no-table` 明显好于 `M5-instance-OBB-no-table`，说明简单 OBB 可能过于粗糙或过于保守。
- 但当前 instance no-table 仍低于 `M5-global ESDF baseline` 的 9 / 20。
- 这不代表 instance-aware 方向错误，更可能说明：
  - 当前实例点云只有可见表面，封闭/补全策略还不够好。
  - no-table 版本改变了规划约束，可能让执行更容易碰到桌面或产生不稳定接触。
  - OBB/voxel 参数还需要调，例如 padding、dilation、min points、workspace crop。
  - M5-global 虽然粗糙，但它保留了更连续的整体几何场，短期内对 cuRobo 可能更友好。

下一步建议：

1. 先可视化 `M5-instance-ESDF-no-table` 与 `M5-global ESDF` 的同 seed 差异。
2. 检查 `object_not_lifted` 的失败视频，判断是碰桌、碰障碍、还是 grasp 候选本身不佳。
3. 给 instance ESDF 加轻量 dilation / padding ablation。
4. 再做 `instance-ESDF + table` 单独分支，判断桌子是否必须回到 collision world。
