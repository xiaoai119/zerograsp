# ManiSkill-cuRobo Real Robot Migration Notes

本文档记录当前 `maniskill_curobo` / `maniskill_codex` 抓取执行链路中用到的 ManiSkill 仿真真值，以及如果迁移到真实机器人时需要替换成哪些真实感知、标定和控制数据。

## 专项文档

- [cuRobo 真机机械臂碰撞球标定步骤](ROBOT_COLLISION_SPHERE_CALIBRATION.md)

## 当前链路中用到的仿真真值

当前系统不是只把 ZeroGrasp 或 GraspNet 的输出直接丢给 cuRobo。实际执行前后还用了大量 ManiSkill 提供的精确状态。这些数据在真机上不能直接获得，需要替换。

### 1. RGB-D 与目标 mask

当前来源：

- `maniskill_codex/zerograsp_inputs.py`
- `maniskill_codex/export_zerograsp_input.py`
- `maniskill_curobo/scripts/execute_curobo_pick.py`

当前使用的仿真数据：

- `obs["sensor_data"][camera]["Color"]`
  - ManiSkill 渲染出的 RGB 图像。
- `obs["sensor_data"][camera]["PositionSegmentation"]`
  - z 通道被用作 depth。
  - segmentation 通道被用作实例/目标 mask。
- ManiSkill actor segmentation id。
  - 当前可以直接知道哪个像素属于目标物体。
  - 也可以根据 `task-target` / `all-objects` / `visible-area` 不同模式导出 mask。

真机替换：

- 真实 RGB-D 相机图像，例如 RealSense、Azure Kinect、工业 RGB-D 相机。
- depth 需要和 RGB 对齐。
- 目标 mask 需要来自真实感知：
  - 语义/实例分割模型，例如 SAM、YOLO-seg、Mask R-CNN。
  - 人工点击目标后用分割模型扩展。
  - 任务系统给定目标类别或目标区域。
  - 连续任务中用 tracker 跟踪目标。

注意：

仿真中的 mask 是完美的；真机 mask 会有遮挡、噪声、漏检和误检。ZeroGrasp/GraspNet 的实际效果会强依赖这个 mask 质量。

### 2. 相机内参和外参

当前来源：

- `env.unwrapped.scene.sensors[camera].camera.get_intrinsic_matrix()`
- `env.unwrapped.scene.sensors[camera].camera.get_model_matrix()`

当前使用的仿真数据：

- 精确 camera intrinsic matrix。
- 精确 camera-to-world pose。
- 通过 ManiSkill 世界坐标和机器人 base 坐标，把抓取位姿从 camera frame 转到 robot base frame。

真机替换：

- 相机内参：
  - 相机标定文件。
  - 或相机 SDK 提供的 intrinsic。
- 相机外参：
  - 手眼标定得到 `T_base_camera`。
  - 如果是 eye-in-hand，需要维护 `T_ee_camera` 并结合当前机器人 FK。
  - 如果是固定外部相机，需要标定 `T_base_camera`。

注意：

这是从感知输出到机器人执行的关键坐标桥。只要外参偏几毫米或几度，抓取点就可能从物体表面偏到桌子或空中。

### 3. 机器人当前状态

当前来源：

- `env.unwrapped.agent.robot.get_qpos()`
- `env.unwrapped.agent.robot.get_pose()`
- ManiSkill active joint names。

当前使用的仿真数据：

- 精确机器人关节角 `q_start`。
- 精确 robot base pose。
- 精确关节顺序和 active joint names。

真机替换：

- 机器人驱动或 ROS/ROS2 joint states。
- 真机 URDF/SRDF 中的关节顺序。
- 真实 robot base frame 定义。
- 如果 robot base 固定在桌面上，应使用标定好的固定 base frame。

注意：

cuRobo 的 `q_start` 必须和 robot config 中的 joint order 一致。真机 joint order、joint limit、home pose、gripper joint 都需要核对。

### 4. hand / TCP 标定

当前来源：

- ManiSkill 的 `panda_hand` link。
- ManiSkill 的 `tcp` link。
- `hand_tcp_transform_in_hand_frame(env)`。

当前使用的仿真数据：

- 精确 hand frame 到 TCP frame 的变换。
- 精确夹爪几何。

真机替换：

- 实际夹爪 TCP 标定。
- 实际 finger 长度、安装偏移、夹爪中心。
- 如果真机夹爪和仿真 Panda gripper 不完全一致，需要同步更新：
  - cuRobo robot config。
  - 抓取执行时的 TCP offset。
  - gripper open/close 宽度映射。

注意：

ZeroGrasp 输出的是抓取几何意义上的 pose，cuRobo 执行的是 robot hand/TCP 的目标 pose。二者之间必须有稳定、准确的 TCP 标定。

### 5. cuRobo world collision model

当前来源：

- `maniskill_curobo/scene_export.py`
- `maniskill_scene_to_curobo_dict(env, world_from_base_matrix, ...)`

当前使用的仿真数据：

- `scene.get_all_actors()`。
- 每个 actor 的精确 pose。
- 每个 actor 的 collision shape。
- 桌子、障碍物、物体等精确碰撞体。
- 根据 ZeroGrasp scene json 和 segmentation id，把目标物体从 cuRobo collision world 中排除。

真机替换：

- 最小版本：
  - 固定桌面 cuboid。
  - 固定 workspace bounds。
  - 固定 robot 周围障碍物。
  - 目标物体不加入 collision world。
- 中等版本：
  - 从 RGB-D 点云分割桌面和障碍物。
  - 将点云拟合成 cuboid 或 mesh。
  - 对非目标物体建近似碰撞体。
- 完整版本：
  - 使用 depth camera 构建 voxel / ESDF / nvblox world。
  - cuRobo 对 robot 与感知到的 3D 世界做碰撞检查。

注意：

这是当前仿真到真机差距最大的一块。现在的实验使用的是 ManiSkill 精确碰撞体；真机一般拿不到每个物体的精确 mesh 和 pose，只能用感知近似。

### 6. 目标物体排除逻辑

当前来源：

- `target_segmentation_ids_from_zerograsp_scene(...)`
- ManiSkill segmentation id。

当前使用的仿真数据：

- 可以准确知道哪个 actor 是目标物体。
- 可以把目标物体从 cuRobo world collision 中排除，避免 planner 把要抓的物体当成障碍物。

真机替换：

- 通过目标 mask 或目标点云区域排除目标物体。
- 如果使用点云建图，需要从 obstacle cloud 中扣掉 target cloud。
- 如果只使用桌面 cuboid，则先不建目标物体碰撞体。

注意：

如果目标物体没有被排除，cuRobo 很可能认为抓取路径撞到了目标物体，从而规划失败。
如果排除过多，真实障碍物又可能被忽略，增加碰撞风险。

### 7. 成功判定和 reward / metric

当前来源：

- `object_lift_trace_sample(env, ...)`
- `compute_object_lift_metrics(...)`

当前使用的仿真数据：

- 直接读取目标物体 pose。
- 直接计算目标物体高度变化。
- 直接判断目标是否被 lift。
- 可以记录物体和 TCP 距离。

真机替换：

- 抬起后用 RGB-D 检查目标是否离开桌面。
- 用目标 mask / tracker 判断目标是否跟随夹爪移动。
- 用 gripper width 判断是否夹住物体。
- 用夹爪电流、力传感器、触觉传感器判断接触。
- 必要时使用外部相机或 AprilTag/marker 做验证。

注意：

真机上的成功判定不再是完美真值。它通常只能是多种信号组合出来的近似判断。

### 8. reset、seed 和 settle

当前来源：

- `env.reset(seed=...)`
- settle steps。

当前使用的仿真数据：

- 可以用 seed 重现同一个场景。
- 可以等待若干 step，让物体在仿真中稳定。
- 可以反复从完全相同的初始状态重跑。

真机替换：

- 真机没有 seed reset。
- 需要人工或自动摆放物体。
- 拍摄前需要等待机械臂、物体和相机画面稳定。
- 如果目标物体被碰动，需要重新感知，而不是假设初始状态不变。

注意：

真机闭环系统需要在每次执行前重新拍照、重新估计目标和场景，而不是依赖一次性初始化。

## 当前代码中哪些模块可以保留

以下部分迁移到真机时大概率可以复用，但输入来源需要替换：

- ZeroGrasp / GraspNet inference 调用。
- grasp candidate 的 top-k 筛选逻辑。
- zero depth / full depth / corrected depth 的候选对比逻辑。
- camera-frame grasp pose 到 base-frame pose 的数学转换。
- cuRobo planner 的调用方式。
- robot config 的大部分结构。

需要重点校验：

- robot config 是否和真机 URDF、joint limits、collision spheres 一致。
- TCP offset 是否和真机夹爪一致。
- gripper width 单位、开合范围、控制接口是否一致。

## 真机迁移建议顺序

### 阶段 1：替换输入，但不执行

目标：

- 用真实 RGB-D 和 mask 跑 ZeroGrasp / GraspNet。
- 用真实标定把抓取 pose 转到 robot base。
- 只可视化结果，不发给机器人执行。

需要完成：

- `real_rgbd_capture`
- `real_target_mask`
- `T_base_camera` 标定读取
- grasp pose 可视化

### 阶段 2：接入 cuRobo planning，但不执行

目标：

- 用真实 joint states 作为 `q_start`。
- 用固定桌面 cuboid 作为 world collision。
- 检查 cuRobo 是否能生成合理轨迹。

需要完成：

- `real_robot_state`
- `real_scene_to_curobo`
- planning-only demo

### 阶段 3：低速真机执行

目标：

- 只执行规划成功且通过安全检查的 top-k 抓取。
- 低速执行 pre-grasp、grasp、lift。
- 人工监控。

需要完成：

- trajectory controller 接口。
- gripper open/close 接口。
- emergency stop / workspace guard。

### 阶段 4：真实成功判定

目标：

- 替换仿真 `object_lift_metrics`。
- 用视觉和夹爪状态判断是否抓起。

需要完成：

- 抬起后 RGB-D 检查。
- gripper width/contact 检查。
- 失败原因分类。

### 阶段 5：更完整的真实碰撞世界

目标：

- 不再只用固定桌面。
- 用真实 depth 点云或 nvblox 建动态 collision world。

需要完成：

- 桌面点云分割。
- 障碍物点云到 cuboid/mesh/ESDF。
- 目标物体点云排除。

## cuRobo world collision model 推进计划

本节专门记录如何把当前依赖 ManiSkill 精确 actor collision shape 的 `scene_source=maniskill`，逐步替换成真机可获得的 collision world。

当前仿真版本的核心问题是：

- 当前 `maniskill_curobo/scene_export.py` 可以直接读取所有 actor 的精确碰撞体。
- 真机没有 actor、segmentation id、精确 mesh pose。
- 真机只能通过标定、RGB-D、点云分割、目标 mask、CAD 先验等方式构造近似碰撞世界。

因此不应该一步到位追求完整 3D 场景重建，而应该从最小可用版本开始，逐层增加真实感知能力。

### M0：保留仿真真值 baseline

目标：

- 保持当前 `scene_source=maniskill` 不变。
- 作为后续所有真实 collision world 的对照组。

输入：

- ManiSkill actor collision shape。
- ManiSkill actor pose。
- ManiSkill segmentation id。

输出：

- 当前格式的 cuRobo `curobo_scene.yml`。

验收：

- 当前 seed1-200 / 其他已有实验可以复现。
- 后续每个新版本都要和 M0 对比 planning failure、grasp failure、lift success。

### M1：固定桌面 + workspace bounds

这是最小真机可用版本。

目标：

- 不再读取 ManiSkill actor collision shape。
- 只给 cuRobo 一个固定桌面 cuboid 和机器人 workspace 边界。
- 目标物体不加入 collision world，避免 planner 把要抓的目标当成障碍物。

输入：

- 手工配置的桌面参数：
  - `table_center_base`
  - `table_size`
  - `table_height`
- 手工配置的 workspace：
  - x/y/z min/max
  - robot base 到桌面的关系

输出：

- `real_scene_static.yml`
- 或运行时生成的 cuRobo world dict。

实现建议：

- 新增 `maniskill_curobo_real/scene_builder.py`。
- 实现 `build_static_table_world(config)`。
- 在 `execute_curobo_pick.py` 或新的 real demo 中支持：
  - `scene_source=static_table`
  - `--real-scene-config path/to/config.yml`

优点：

- 实现最快。
- 最接近真机早期部署方式。
- 可以先验证 grasp pose、TCP、cuRobo planning 和执行链路是否通。

缺点：

- 看不到桌面上的非目标障碍物。
- 如果目标附近有杂物，planner 不会避障。
- 桌面高度标定错误会直接导致碰桌或保守规划失败。

验收：

- 在仿真中禁用 ManiSkill actor collision，只用固定桌面 cuboid 重跑 PickSingle。
- 比较 `scene_source=maniskill` 和 `scene_source=static_table`：
  - pre-grasp planning success。
  - grasp planning success。
  - lift success。
  - 是否出现碰桌。

### M2：由目标 mask 排除目标区域

目标：

- 为后续点云建障碍物做准备。
- 明确哪些点属于目标物体，不能作为障碍物加入 cuRobo world。

输入：

- RGB-D 图像。
- 目标 mask。
- camera intrinsics。
- `T_base_camera`。

输出：

- `target_cloud_base`
- `non_target_cloud_base`

实现建议：

- 新增 `depth_to_pointcloud(...)`。
- 新增 `split_target_and_obstacle_cloud(rgbd, target_mask, intrinsics, T_base_camera)`。
- 当前阶段可以只保存点云，不一定用于规划。

优点：

- 可以验证真机感知链路里的目标分割质量。
- 后续 M3/M4 都依赖这个 target exclusion。

缺点：

- mask 错误会导致两类问题：
  - 目标没有被完全排除，planner 把目标当障碍物。
  - 障碍物被误排除，planner 可能撞到真实物体。

验收：

- 在仿真中用 `PositionSegmentation` 生成“完美 mask”先跑通。
- 再用真实分割模型或弱化 mask 做鲁棒性测试。
- 可视化 target cloud / non-target cloud。

### M3：oracle instance 点云 + 非目标物体 cuboid

目标：

- 先在仿真里使用 ManiSkill 的 instance segmentation，把每个物体的点云单独分出来。
- 对每个非目标物体拟合一个 cuboid，喂给 cuRobo。
- 排除目标物体本身，避免 planner 把要抓的物体当成障碍物。
- 诊断真实部署时需要的“实例级感知”能力。

输入：

- RGB-D 图像。
- 目标 mask。
- instance segmentation ids。
- scene object records。
- `T_base_camera`。
- workspace crop 范围。

处理流程：

1. depth 转 base frame 点云。
2. workspace crop。
3. 用非目标点云估计桌面高度，生成 table cuboid。
4. 读取每个物体的 instance id。
5. 跳过目标物体 instance。
6. 每个非目标物体 instance 单独拟合 AABB cuboid。
7. cuboid 默认直接使用该 instance 点云的 xyz min/max AABB，不额外缩放。
8. 输出 cuRobo cuboid world。

输出：

- `curobo_scene.yml`
- debug 可视化：
  - 原始点云。
  - 目标点云。
  - 按 instance 区分的物体点云。
  - 每个 instance 的障碍物 cuboids。

优点：

- 仍然是 cuRobo 很容易处理的 cuboid world。
- 比 M1/M2 更接近真实障碍物布局。
- 相比旧的非目标点云聚类，不容易把多个物体合成一个巨大 cuboid。
- 计算量可控，适合早期真机在线规划。

缺点：

- 当前实现依赖 ManiSkill 的完美 instance segmentation，是仿真 oracle。
- 真机不能直接拿到这个信号，需要由实例分割 / 物体检测 / 语义分割模块替代。
- AABB 仍然比较粗，细长物体和斜放物体可能被包得偏大。
- 可以通过负 padding 手动缩小 cuboid 以降低过度保守，但默认不缩小，避免低估真实碰撞体。

- 对透明/反光/黑色物体不稳定。
- cuboid 会偏保守，可能挡掉一些本来可行的轨迹。
- 复杂形状用 cuboid 近似会粗糙。

验收：

- 在 PickSingle 中模拟真实感知，只用 depth 点云重建桌面和障碍物。
- 和 M1 对比：
  - 是否减少碰撞风险。
  - 是否增加 planning failed。
  - 是否影响抓取成功率。

### M4：三阶段精细碰撞世界

M4 保持 M3 的数据来源不变：

- 使用同一张稳定后的 RGB-D。
- 使用 oracle instance segmentation 区分不同物体。
- 排除目标物体，只把桌面和非目标物体放进 cuRobo world。
- 不读取 ManiSkill actor collision mesh、精确物体位姿或 CAD 真值。

这样 M4-A/B/C 与 M3 的差别只来自碰撞几何表达，能够公平判断更精细的
world model 带来了多少成功率变化和额外耗时。

#### M4-A：每个 instance 的 yaw OBB

步骤：

1. 取得每个非目标 instance 的可见点云。
2. 在桌面 xy 平面做 PCA，估计物体主要朝向。
3. 沿该朝向拟合可旋转长方体，z 方向仍使用点云的 min/max。
4. 将 yaw OBB 作为 cuRobo cuboid obstacle。

相对 M3 的变化：

- M3 使用与机器人 base 轴对齐的 AABB。
- M4-A 允许长方体绕 z 轴旋转，更贴合斜放的细长物体。
- 碰撞查询仍然是 cuboid，预计运行代价只小幅增加。

预期：

- 减少 AABB 空角造成的过度保守规划。
- 成功率应不低于 M3，单次总耗时接近 M3。

#### M4-B：每个 instance 的凸包 mesh

步骤：

1. 取得每个非目标 instance 的可见点云并下采样。
2. 使用三维 convex hull 生成闭合三角网格。
3. 将每个 instance 的 hull 作为独立 cuRobo mesh obstacle。
4. 凸包失败或点云退化时回退到 M4-A yaw OBB。

相对 M4-A 的变化：

- 不再用一个长方体覆盖物体。
- 斜面、圆柱和不规则外形可以保留更多几何细节。
- mesh 碰撞缓存和查询比 cuboid 更贵。

预期：

- 进一步减少虚假的碰撞体积。
- 规划可行率可能提高，但 planner 构建和碰撞查询耗时会增加。

#### M4-C：工作区 voxel ESDF

步骤：

1. 取得每个非目标 instance 的可见点云。
2. 先生成闭合凸包，避免把单视角表面点误当成没有内部体积的薄壳。
3. 在固定工作区网格内体素化全部非目标物体。
4. 对占用网格计算 signed distance field：
   - 物体内部为负距离。
   - 自由空间为正距离。
5. 将 ESDF 作为 cuRobo `VoxelGrid`，桌面仍使用独立 cuboid。

第一版使用单帧、10 mm voxel，目标是验证原生 cuRobo ESDF 链路。它还不是
多帧 nvblox 地图；多帧融合和在线更新仍属于 M5。

预期：

- 几何分辨率固定，不随 mesh 三角形数量变化。
- 更接近未来真机在线地图的碰撞接口。
- planner 初始化、显存占用和碰撞查询预计是三阶段中最贵的。

#### 固定评测协议

每个版本都使用以下完全相同的设置：

- 环境：`PickClutterYCB-v1`。
- seeds：`1-200`。
- 抓取候选：复用现有 full-depth ZeroGrasp 输出。
- ZeroGrasp 候选数：`top-k=20`。
- 抓取深度：`scale=1.0`，保留现有自动 depth fallback。
- 拍摄前稳定：`20 steps`。
- 相机：`eye=[-0.2, 0.0, 0.27]`，`target=[0.05, 0.0, 0.08]`。
- M3 基线：oracle instance、AABB、`padding=0`。
- 默认不录视频，避免视频编码影响计时。

基线结果使用已经完成的正式 M3：

- 完成：`197/200`。
- lift 成功：`101/197 = 51.3%`。
- 缺少 ZeroGrasp candidate：seed `1、40、169`。

每个 M4 版本必须报告：

- processed / complete / lift success。
- pre、grasp、lift planning failed 数量。
- object not lifted 和 execution failed 数量。
- 单 seed 总耗时的 mean / median / p95。
- 相对 M3 的 lift 成功率变化，单位为百分点。
- 相对 M3 的平均耗时倍率。
- M3 成功但该版本失败、M3 失败但该版本成功的 seed。

执行顺序：

1. 几何单元测试。
2. 每个版本先跑代表性 smoke seeds。
3. smoke 通过后分别跑 seed `1-200`。
4. 生成统一的 `m3_vs_m4_comparison.json`、CSV 和 Markdown 报告。

验收原则：

- M4-A 若明显慢于 M3 或成功率下降，先检查 OBB 姿态和点云坐标系。
- M4-B 若 mesh 导致大量 planning failed，检查 hull 是否闭合、面索引和 mesh pose。
- M4-C 若出现全失败，先用已知障碍物做 ESDF 正负号测试，不直接归因于规划器。
- 只有在同 candidate、同抓取深度、同 workspace、同执行参数下才比较性能衰减。

#### M0/M3/M4 seed1-200 实测结果

完成日期：2026-06-12。

为了公平比较运行时间，M3 也使用与 M4 相同的持久化 ManiSkill 环境、持久化
cuRobo planner 和 `update_world` 方式重新运行。优化后的 M3 仍然得到与历史正式
基线完全相同的 `101/197` 次 lift 成功，因此没有改变成功率基线。

M0 是 ManiSkill 仿真真值碰撞世界 baseline。M0、M3 和三个 M4 版本都使用同一批
ZeroGrasp full-depth top-k 候选、相同 depth fallback、相机、settle steps、
workspace 和执行参数。seed `1、40、169` 缺少可复用的 ZeroGrasp candidate，
因此每个版本实际完成 `197/200`。

| 版本 | 碰撞表达 | Lift 成功 | 相对 M3 | 改善 / 退化 seed | 端到端平均耗时 | 相对 M3 耗时 |
|---|---|---:|---:|---:|---:|---:|
| M0 | ManiSkill truth collision | `94/197 = 47.72%` | `-3.55 pp` | - | `6.40 s` | `0.79x` |
| M3 | instance AABB | `101/197 = 51.27%` | baseline | - | `8.13 s` | `1.00x` |
| M4-A | yaw OBB | `103/197 = 52.28%` | `+1.02 pp` | `8 / 6` | `7.81 s` | `0.96x` |
| M4-B | convex hull mesh | `8/197 = 4.06%` | `-47.21 pp` | `0 / 93` | `16.01 s` | `1.97x` |
| M4-C | 10 mm voxel ESDF | `111/197 = 56.35%` | `+5.08 pp` | `20 / 10` | `7.75 s` | `0.95x` |

这里的端到端耗时包含该 seed 的碰撞世界构建和抓取规划/执行，不包含已经
离线复用的 ZeroGrasp 推理。M0 不需要构建点云 world，因此比 M3/M4 更快。
平均耗时也会受失败类型影响：M4-B 经常依次测试 top-20 候选后才宣布失败，
因此既慢又没有执行抓取。

失败分布：

| 版本 | Candidate selection failed | Grasp planning failed | Lift planning failed | Object not lifted | Execution failed |
|---|---:|---:|---:|---:|---:|
| M0 | 39 | 2 | 0 | 44 | 18 |
| M3 | 49 | 2 | 0 | 44 | 1 |
| M4-A | 47 | 0 | 0 | 46 | 1 |
| M4-B | 184 | 1 | 0 | 3 | 1 |
| M4-C | 24 | 3 | 1 | 57 | 1 |

结论：

- **M0 不是当前成功率上限。** ManiSkill 真值碰撞世界在这批候选上只有
  `94/197 = 47.72%`，低于 M3/M4-C。原因是 M0 对仿真真值碰撞更硬，
  candidate selection 和 execution failed 更多；更真实的碰撞世界不一定给
  当前 ZeroGrasp 候选带来更高执行成功率。
- **M4-C 是当前最好的 M4 版本。** 它把 lift 成功率从 `51.27%` 提高到
  `56.35%`，净增加 10 个成功 seed，端到端耗时没有衰减。
- **M4-A 可以作为低风险 fallback。** 它保持 cuboid 查询，成功率小幅提升，
  运行成本与 M3 基本相同。
- **M4-B 不应直接作为默认 world model。** mesh 文件经过 watertight、winding、
  volume 和坐标范围检查，但硬凸包使 184 个 seed 在 top-k candidate selection
  阶段失败。问题不是 mesh 文件损坏，而是闭合凸包作为硬碰撞体对贴近物体的抓取
  过于保守。
- M4-C 仍使用 oracle instance segmentation 和单帧点云，不等于已经完成真机
  感知迁移。下一步应把 instance 输入换成真实实例分割，并进入 M5 多帧
  nvblox/ESDF 更新。

#### 暂存分支：多帧 / 多视角融合 ESDF

该方向记录为后续独立实验分支，当前不进入 M4-C 主线实现和基准结果。

动机：

- 当前 M4-C 只使用单个相机视角的可见表面点云。
- 被遮挡的背面无法直接观测，闭合凸包只是几何补全近似。
- 10 mm 体素会抹掉小物体、球体和积木的局部细节。
- 单视角碰撞体可能偏小，从而放行仿真中成功、但真机上不够安全的轨迹。

分支方案：

1. 从固定相机的多个视角，或 eye-in-hand 相机的多个机器人位姿采集 RGB-D。
2. 使用每帧 `T_base_camera` 将深度点云统一变换到 robot base frame。
3. 对每帧应用目标 mask，持续排除目标物体点云。
4. 融合非目标物体点云或 TSDF，再计算统一 ESDF。
5. 对比单帧 M4-C：
   - 碰撞体完整度。
   - planning success。
   - lift success。
   - 碰撞漏检。
   - 建图与规划耗时。

启用该分支前需要先解决：

- 相机位姿标定误差和多帧配准误差。
- 物体在采集过程中移动造成的重影。
- 每帧目标 mask 不一致导致目标残留在碰撞地图中。
- 地图清理、更新频率和显存占用。

当前主线保持不变：

- 单帧 RGB-D。
- oracle instance segmentation。
- 排除目标 instance。
- 非目标 instance 闭合凸包体素化。
- 10 mm 原生 cuRobo ESDF。

主线下一步先继续检查单帧 M4-C 的几何质量、碰撞安全性和成功率来源，不因为
多视角分支而修改当前 `111/197` 的正式基准。

结果文件：

- 公平 M3 复跑：
  `runs/pickclutter_full_depth_m3_optimized_seed1_200/records.jsonl`
- M4-A/B/C 原始记录：
  `runs/pickclutter_full_depth_m4abc_seed1_200/records.jsonl`
- 汇总 JSON：
  `runs/pickclutter_full_depth_m4abc_seed1_200/comparison/m3_vs_m4_comparison.json`
- 逐 seed CSV：
  `runs/pickclutter_full_depth_m4abc_seed1_200/comparison/m3_vs_m4_per_seed.csv`
- 简表：
  `runs/pickclutter_full_depth_m4abc_seed1_200/comparison/m3_vs_m4_comparison.md`

### M3-ZG / M4-ZG：使用 ZeroGrasp 重建点云复刻 M3-M4

这条实验线从 M3 开始，不再使用 ManiSkill depth 反投影得到的物体表面点云，
而是使用 ZeroGrasp 内部重建出的每个物体完整表面点云。碰撞几何表达、抓取候选、
执行参数和结果产物保持与现有 M3/M4-A/B/C 一致。

固定不变的部分：

- 仍复用同一批 full-depth ZeroGrasp top-k 抓取候选。
- 仍使用相同相机、20-step settle、depth fallback 和执行参数。
- M3 仍拟合每个非目标实例的 AABB。
- M4-A 仍拟合 yaw OBB。
- M4-B 仍生成闭合凸包 mesh。
- M4-C 仍生成 10 mm voxel ESDF。
- 输出仍包含逐 seed scene、metadata、debug 框图、run manifest、records、
  summary 和 M3-vs-M4 对比表。

改变的部分：

1. 使用 `all-objects` mask 把场景中每个可见物体作为独立 ZeroGrasp 实例。
2. 保存 ZeroGrasp `ObjectGraspResult.point_cloud` 和 normals。
3. 将点云从 OpenCV camera frame 的毫米单位转换到 robot base frame 的米单位。
4. 根据 `camera.json` 中的 `is_task_target` 排除目标物体。
5. 使用其余实例的 ZeroGrasp 重建表面生成 M3/M4 碰撞几何。
6. 桌面仍使用同一固定 table cuboid；不从 ZeroGrasp 重建桌面。

需要明确的限制：

- 点云几何来自 ZeroGrasp，不再来自 ManiSkill depth 点云。
- 当前 `all-objects` 实例 mask 仍由 ManiSkill oracle segmentation 生成，因此
  这是“替换几何重建来源”的公平实验，还不是完整真机感知实验。
- ZeroGrasp 对遮挡物体的重建质量会直接影响包围盒、mesh 和 ESDF。

先生成 seed `1-200` 的多实例重建：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python \
  -m maniskill_curobo_real.generate_zerograsp_reconstructions \
  --env-id PickClutterYCB-v1 \
  --seed-start 1 \
  --seed-end 200 \
  --settle-before-export-steps 20 \
  --output-root \
  maniskill_curobo_real/runs/pickclutter_zerograsp_reconstructions_seed1_200
```

再运行 M3/M4-A/B/C：

```bash
maniskill_curobo/envs/maniskill_curobo/bin/python \
  -m maniskill_curobo_real.run_world_collision_stages \
  --env-id PickClutterYCB-v1 \
  --seed-start 1 \
  --seed-end 200 \
  --stages m3 m4a m4b m4c \
  --point-cloud-source zerograsp_reconstruction \
  --zerograsp-reconstruction-root \
  maniskill_curobo_real/runs/pickclutter_zerograsp_reconstructions_seed1_200 \
  --reuse-candidate-root \
  maniskill_curobo_real/runs/m4_full_depth_candidates_seed1_200 \
  --output-root \
  maniskill_curobo_real/runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200
```

最后可继续复用 `summarize_m4_benchmark.py`，把同一个 `records.jsonl`
同时作为 M3 和 M4 输入，生成与原实验同名的 JSON、CSV 和 Markdown 对比产物。

#### seed1-200 正式结果

本轮已经完成：

- 场景：`PickClutterYCB-v1`
- seed：`1-200`
- 重建输入：settle 20 step 后的 RGB-D 与 `all-objects` oracle instance mask
- 重建完成：`200/200`
- 重建实例：`1153`
- 重建点总数：`8,409,921`
- 抓取候选、top-k、depth fallback 和执行协议与 ManiSkill 点云实验一致
- seed `1`、`40`、`169` 缺少可复用抓取候选，因此每个阶段实际完成 `197` 个

ZeroGrasp 重建点云内部比较：

| Stage | Lift success | 相对 M3-ZG | 主要结果 |
|---|---:|---:|---|
| M3-ZG AABB | 91/197（46.19%） | baseline | 完整重建表面的轴对齐范围会放大几何误差 |
| M4A-ZG yaw OBB | 100/197（50.76%） | +4.57 pp | 比 AABB 更贴合物体平面朝向 |
| M4B-ZG convex mesh | 12/197（6.09%） | -40.10 pp | mesh 碰撞世界仍过于保守，不适合作为主线 |
| M4C-ZG voxel ESDF | 108/197（54.82%） | +8.63 pp | 本轮最佳 ZeroGrasp 重建点云版本 |

与原 ManiSkill depth 点云逐 seed 公平比较：

| Stage | ManiSkill depth | ZeroGrasp reconstruction | 差值 |
|---|---:|---:|---:|
| M3 | 101/197（51.27%） | 91/197（46.19%） | -5.08 pp |
| M4A | 103/197（52.28%） | 100/197（50.76%） | -1.52 pp |
| M4B | 8/197（4.06%） | 12/197（6.09%） | +2.03 pp |
| M4C | 111/197（56.35%） | 108/197（54.82%） | -1.52 pp |

这说明 ZeroGrasp 重建点云已经可以替换 ManiSkill depth 点云来构造 M3/M4
碰撞世界。M4C 的 lift success 只损失 3 个 seed，性能已经非常接近原实验。
M3 下降更明显，原因是完整表面的极值点会直接扩大 AABB；M4A 和 M4C
对点云重建误差更稳健。M4B 虽有小幅提升，但绝对成功率仍太低。

当前仍不是完整真机链路：实例划分和目标标识继续使用 ManiSkill oracle
segmentation。几何点的位置与表面形状来自 ZeroGrasp 重建，不使用
ManiSkill depth 点云或 actor collision shape。

正式产物：

- 多实例重建：
  `runs/pickclutter_zerograsp_reconstructions_seed1_200`
- M3/M4-A/B/C 场景、框图和执行结果：
  `runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200`
- ZeroGrasp 点云内部 M3-vs-M4：
  `runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200/comparison`
- ManiSkill-vs-ZeroGrasp 逐 seed 比较：
  `runs/pickclutter_zerograsp_reconstruction_m3_m4abc_seed1_200/source_comparison`

### M5：nvblox / ESDF 在线碰撞世界

这是完整版本。

目标：

- 使用 depth camera 持续融合场景。
- 建立 nvblox / ESDF collision world。
- cuRobo 直接在动态更新的 3D world 中做碰撞查询。

输入：

- 连续 RGB-D / depth frames。
- camera pose stream：
  - 固定相机：固定 `T_base_camera`。
  - eye-in-hand：机器人 FK + `T_ee_camera`。
- target mask / target cloud exclusion。

输出：

- 在线更新的 nvblox / ESDF world。
- cuRobo motion generation 使用该 world collision。

优点：

- 最接近真机长期可用形态。
- 可以处理动态障碍物和复杂工作台。
- 不再依赖物体 CAD 或精确 pose。

缺点：

- 工程成本最高。
- 需要处理相机延迟、深度噪声、地图清理、目标物体排除。
- 需要严格做安全验证。

验收：

- planning-only：
  - 多帧融合后规划。
  - 移动物体后 world 更新是否生效。
- low-speed execution：
  - 低速执行。
  - 观察是否绕开非目标障碍物。
- stress test：
  - 遮挡。
  - 深度缺失。
  - 目标附近杂物。

## 推荐实现顺序

短期应该先做 M1 和 M2：

1. `static_table` world builder。
2. 用仿真复现实验，但禁止读取 actor collision shape。
3. 加入真实格式的 depth-to-pointcloud 和 target cloud exclusion。
4. 评估 M1 相比当前 ManiSkill 真值 world 的损失。

中期做 M3：

1. 从点云拟合桌面。
2. 先使用 oracle instance segmentation 生成每物体 cuboid obstacles。
3. 默认使用每个 instance 的原始 AABB，必要时再实验性调整 padding。
4. 再把 oracle instance segmentation 替换成真实实例分割输出。
5. 在仿真里模拟真实感知噪声，检查 planning 是否过度保守。

长期再做 M4/M5：

1. mesh / voxel world。
2. nvblox / ESDF 在线更新。
3. 真机闭环安全验证。

## 第一版建议验收指标

M1/M2 第一版不追求抓取成功率超过当前 ManiSkill 真值版本，而是先看链路是否可替换：

- `planning_success_rate`
- `pregrasp_planning_failed_count`
- `grasp_planning_failed_count`
- `table_collision_like_failure_count`
- `object_not_lifted_count`
- `lift_success_rate`
- 平均规划耗时
- 平均完整 episode 耗时

如果 `static_table` world 的 lift success 接近当前 `maniskill` world，说明当前任务主要受抓取位姿影响，碰撞世界可以先简化。
如果 planning failed 明显增加，说明当前 cuRobo 对桌面/障碍物建模很敏感，需要尽快推进 M3。

## 总结

当前仿真系统里，ZeroGrasp/GraspNet 输出只是抓取候选的一部分。真正让系统能稳定跑起来的，是 ManiSkill 提供的一整套完美状态：

- 完美 RGB-D。
- 完美目标 mask。
- 完美 camera pose。
- 完美 robot state。
- 完美 scene collision geometry。
- 完美 target object lift metric。

上真机时，cuRobo 本身可以继续使用，但这些仿真真值都必须替换成真实感知、真实标定、真实机器人状态和近似碰撞建图。
