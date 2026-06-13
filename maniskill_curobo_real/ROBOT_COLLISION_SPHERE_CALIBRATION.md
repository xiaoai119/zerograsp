# cuRobo 真机机械臂碰撞球标定步骤

本文档说明如何为真实机械臂建立、验证和维护 cuRobo 使用的机器人碰撞球模型。

碰撞球不是由相机在每次规划时重新生成的。它们固定在机器人各个 link 的局部坐标系中，cuRobo 根据实时关节角和正向运动学更新球在机器人基座坐标系中的位置。

一个球通常表示为：

```yaml
collision_spheres:
  wrist_link:
    - center: [0.000, 0.000, 0.050]
      radius: 0.040
```

其中 `center` 是球心在 `wrist_link` 局部坐标系中的位置，`radius` 是半径，单位均为米。

## 1. 标定目标

标定完成后应满足：

1. 碰撞球完整覆盖机械臂、夹爪和随机器人运动的附加硬件。
2. 球体不能存在明显漏覆盖，尤其是肘部、手腕、夹爪和 link 连接处。
3. 球体不能无意义地伸入安装台、桌面或相邻 link。
4. cuRobo 使用的关节顺序、关节零位、link 坐标系和真机一致。
5. 自碰撞忽略列表不会屏蔽真实可能发生的碰撞。
6. 世界碰撞检查保留足够的测量和控制误差余量。

cuRobo 碰撞模型属于规划层模型，不能替代真机的急停、限速、力矩监控、安全围栏和控制器安全功能。

## 2. 所需输入

开始标定前准备以下文件和信息：

- 真机型号和具体硬件版本。
- 厂商提供的 URDF 或 xacro。
- URDF 引用的 STL、DAE 或 OBJ 网格。
- 真实关节名称、顺序、轴方向、限位和零位。
- 机器人基座坐标系定义。
- 实际安装的夹爪及其 URDF/网格。
- 实际 TCP 位置和姿态。
- 腕部相机、转接板、工具和保护罩的网格或测量尺寸。
- 可能进入运动空间的粗线缆和软管信息。

当前工程默认使用：

```text
maniskill_curobo/external/curobo/curobo/content/configs/robot/franka.yml
```

如果真机不是 Franka，不能沿用这个文件中的碰撞球。应为真实机器人生成独立配置，例如：

```text
maniskill_curobo_real/config/real_robot.yml
```

## 3. 检查 URDF 和真机一致性

先检查运动学，再进行球体拟合。错误的 URDF 无法通过增加安全余量修复。

### 3.1 检查尺寸和网格

- URDF 和网格统一使用米。
- 检查 mesh 的 `scale`。
- 检查视觉网格和碰撞网格是否对应真实外壳。
- 检查夹爪、相机和转接件是否缺失。
- 检查基座安装方向和真实设备一致。

### 3.2 检查运动学

- 比较真机驱动返回的 joint names 和 URDF。
- 在 home pose 下比较真机和 URDF 的关节角。
- 逐关节小角度运动，检查正负方向是否一致。
- 检查关节上下限。
- 使用 FK 检查末端位置和真实 TCP 是否一致。

### 3.3 标定 TCP

ZeroGrasp/GraspNet 输出的是抓取几何位姿，cuRobo 控制的是机器人配置中的末端 link。必须获得：

```text
T_ee_tcp
```

其中 `ee` 是 cuRobo 的末端 link，`tcp` 是实际抓取中心。更换夹爪、指尖或转接板后必须重新检查该变换。

## 4. 从 URDF 自动生成初始碰撞球

本仓库中的 cuRobo 提供 `RobotBuilder`，可以从 URDF link collision mesh 自动拟合碰撞球，并生成 self-collision ignore 配置。

在可运行当前 cuRobo 的 Python 环境中执行：

```bash
python -m curobo.examples.getting_started.build_robot_model \
  --urdf /absolute/path/to/real_robot.urdf \
  --asset-path /absolute/path/to/robot_assets \
  --output maniskill_curobo_real/config/real_robot.yml \
  --compute-metrics \
  --visualize
```

推荐使用默认的 MorphIt 拟合方法。若覆盖不足，可增加球密度：

```bash
python -m curobo.examples.getting_started.build_robot_model \
  --urdf /absolute/path/to/real_robot.urdf \
  --asset-path /absolute/path/to/robot_assets \
  --output maniskill_curobo_real/config/real_robot.yml \
  --sphere-density 2.0 \
  --compute-metrics \
  --visualize
```

如果基座球伸入安装平面，可以对基座 link 使用裁剪平面：

```bash
--clip-link base_link z 0.0
```

裁剪坐标和方向位于该 link 的局部坐标系中，使用前必须检查真实安装方向，不能直接照搬示例。

如果仅需重新拟合某个 link：

```bash
python -m curobo.examples.getting_started.build_robot_model \
  --edit-config maniskill_curobo_real/config/real_robot.yml \
  --refit-link wrist_link \
  --sphere-density 2.0
```

## 5. 检查自动拟合指标

启用 `--compute-metrics` 后，重点检查：

- `coverage`：网格内部被球覆盖的比例。
- `protrusion`：球体伸出网格的比例。
- `protrusion_dist_p95`：球体伸出表面的 95 分位距离。
- `surface_gap_p95`：网格表面到最近球表面的 95 分位间隙。
- `max_uncovered_gap`：最大的表面漏覆盖距离。
- `volume_ratio`：球体总体积与 mesh 体积的比例。

判断原则：

- 优先消除漏覆盖，而不是单纯追求更少的球。
- `max_uncovered_gap` 较大的 link 必须可视化检查。
- 对细长、薄片或中空结构，自动拟合结果可能不可靠。
- 球体数量过少会漏覆盖，过多会增加碰撞计算和维护成本。
- 明显 protrusion 会造成规划过度保守，甚至让机器人一直处于碰撞状态。

指标只是筛查工具，不能替代可视化和真机验证。

## 6. 可视化检查

### 6.1 静态检查

在可视化工具中同时显示：

- URDF collision mesh。
- collision spheres。
- link 坐标轴。
- TCP 坐标轴。
- 基座安装平面。

逐个 link 检查：

- 球体是否覆盖真实外轮廓。
- link 连接处是否存在空隙。
- 球体是否穿入相邻 link。
- 基座球是否穿入地面或安装台。
- 夹爪指尖、腕部相机和转接板是否缺失。

### 6.2 多姿态检查

至少检查以下姿态：

- home pose。
- 每个关节的正负极限附近。
- 肘部完全折叠。
- 手腕翻转。
- 夹爪完全张开和闭合。
- 工具朝下、水平和朝上的典型工作姿态。

现有脚本：

```text
maniskill_curobo_real/visualize_robot_collision_spheres_once.py
```

当前脚本主要用于 ManiSkill Franka 的一次性可视化。接入其他真机时，需要替换机器人模型、link 名称和 `--robot-config-path`，不能只把 Franka 球显示在另一台机械臂上。

## 7. 补充自动拟合遗漏的硬件

以下结构经常不在厂商基础 URDF 中：

- 更换后的夹爪。
- 腕部 RGB-D 相机。
- 相机支架和转接板。
- 长工具、吸盘和电批。
- 粗线缆、气管和保护套。

处理顺序：

1. 优先把刚性结构加入 URDF，并为其建立独立 link。
2. 对新增 link 自动拟合碰撞球。
3. 对无法准确建模的软结构使用保守近似球。
4. 不要使用一个巨大球包住整个腕部，这会严重降低可规划空间。

夹爪手指应跟随各自关节运动。不要把张开状态下的两个手指合并成一个固定在手掌上的大球。

抓取物体后，还应将被抓物体作为 attached object 加入碰撞检查，否则 cuRobo 只会避免机械臂碰撞，不会避免手中物体碰撞环境。

## 8. 使用真机点云验证

CAD/URDF 用于生成初始碰撞球，真实点云用于检查碰撞球是否覆盖实际机器人。

### 8.1 前置标定

必须先完成：

- 相机内参标定。
- 深度尺度检查。
- 固定相机的 `T_base_camera` 标定，或眼在手相机的 `T_ee_camera` 标定。
- 时间同步和关节状态同步。

外参错误会被误判为碰撞球错误，因此两者不能同时盲调。

### 8.2 采集数据

在多个关节姿态下保存：

- 时间戳。
- 当前关节角 `q`。
- RGB 和 depth。
- 相机内外参。
- 转换到 robot base frame 的点云。
- 当前 cuRobo 球体在 base frame 中的球心和半径。

采集姿态应覆盖工作空间，而不是只采 home pose。

### 8.3 计算覆盖误差

对于真实机器人表面点 `p`，计算其到所有碰撞球的最小带符号间距：

```text
gap(p) = min_i(||p - c_i|| - r_i)
```

其中：

- `c_i` 是第 `i` 个球在 base frame 中的球心。
- `r_i` 是球半径。
- `gap > 0` 表示真实表面露在所有球之外。
- `gap <= 0` 表示该点被至少一个球覆盖。

需要统计：

- 正 gap 点的数量和比例。
- gap 的均值、95 分位和最大值。
- 每个 link 的漏覆盖区域。
- 负 gap 绝对值很大的区域，即过度膨胀区域。

真机点云会包含遮挡和噪声。验证时应先去除桌面和环境点，并尽可能获得机器人实例 mask。单视角看不到的背面不能被视为已经验证。

## 9. 设置碰撞安全余量

碰撞球半径应优先表达真实机器人几何，额外不确定性通过安全余量处理。

cuRobo 配置中常见参数：

```yaml
collision_sphere_buffer: 0.005

self_collision_buffer:
  wrist_link: 0.005
  hand_link: 0.008
```

含义：

- `collision_sphere_buffer`：机器人与世界碰撞检查使用的额外膨胀。
- `self_collision_buffer`：机器人自身 link 之间碰撞检查使用的额外膨胀。

安全余量至少应覆盖：

- URDF/CAD 尺寸误差。
- 关节零位和编码器误差。
- 控制跟踪误差。
- 基座安装误差。
- 相机外参和深度误差。
- 世界碰撞模型误差。
- 线缆和软性结构的摆动范围。

低速实验阶段可以从约 `5-10 mm` 的保守量级开始，再根据真实误差和规划可行率调整。这不是通用安全标准，最终数值需要结合设备精度、速度、负载和现场风险评估确定。

不要通过把所有球无限放大来获得“安全”。过大的球会导致：

- IK 无解。
- pre-grasp 或 grasp planning failed。
- 狭窄区域无法通过。
- 机器人在初始姿态就被判断为碰撞。

## 10. 标定 self-collision ignore

RobotBuilder 可以通过采样关节配置生成 `self_collision_ignore`，但自动结果必须人工复核。

原则：

- 相邻且设计上允许接触的 link 可以忽略。
- 永远不可能接触的 link 可以降低检查开销。
- 任何真实可能碰撞的 link pair 都不能加入 ignore。
- 新增相机、工具或夹爪后需要重新生成并验证。

验证方式：

1. 在完整关节范围内进行大量随机采样。
2. 单独检查所有关节极限附近。
3. 检查肘部折叠和手腕翻转等高风险姿态。
4. 输出发生最小距离的 link pair。
5. 对可疑 ignore pair 取消忽略并重新测试。

不要仅因为随机采样中没有观察到碰撞，就认定某一对 link 永远安全。

## 11. 真机低速验收

真机验收应分阶段进行。

### 11.1 Planning-only

- 使用真实 joint states 作为 `q_start`。
- 只生成轨迹，不向控制器发送。
- 可视化完整轨迹和碰撞球扫过区域。
- 检查初始状态是否被误判为碰撞。

### 11.2 空场景低速运行

- 清空机器人周围可移动障碍物。
- 设置低速度、低加速度和独立急停。
- 运行覆盖工作空间的无负载轨迹。
- 检查实际机器人与球体可视化是否同步。

### 11.3 已知障碍物验证

使用位置和尺寸已知的桌面、墙面或标定块：

- 将障碍物加入 cuRobo world。
- 先规划不同最小间距的轨迹。
- 对照测量的真实间距和 cuRobo 预测间距。
- 验证整条轨迹，而不是只检查起点和终点。
- 不以真实碰撞作为测试手段。

### 11.4 抓取状态验证

分别验证：

- 夹爪张开。
- 夹爪闭合。
- 空载。
- 带典型尺寸 attached object。

## 12. 建议验收记录

每个碰撞球配置应保存：

- 机器人型号和序列版本。
- URDF、mesh 和 cuRobo YAML 的 Git commit。
- 夹爪、相机、转接件和工具清单。
- TCP 标定结果。
- 自动拟合参数和随机种子。
- 每个 link 的拟合指标。
- 球体可视化截图或视频。
- 真机点云覆盖误差报告。
- self-collision 测试报告。
- 世界障碍物低速验证结果。
- 审核人和日期。

建议目录：

```text
maniskill_curobo_real/
  config/
    real_robot.yml
  calibration/
    robot_description/
    sphere_fit_metrics.json
    pointcloud_validation/
    self_collision_validation/
    reports/
```

## 13. 必须重新标定的情况

发生以下变化后，不应继续直接使用旧配置：

- 更换机械臂型号或硬件版本。
- 更换夹爪、指尖或工具。
- 增加或移动腕部相机。
- 修改相机支架或转接板。
- 改变粗线缆和气管布置。
- 修改 URDF link frame、joint origin 或 mesh scale。
- 重新安装机器人基座。
- 发现真实碰撞、漏检或持续误报。

## 14. 最终检查清单

- [ ] 使用的不是默认 Franka 配置，而是真机对应配置。
- [ ] URDF 尺寸、关节方向、零位和限位已核对。
- [ ] TCP 已标定。
- [ ] 夹爪、相机、支架和工具已加入碰撞模型。
- [ ] 所有球心都定义在正确的 link 局部坐标系中。
- [ ] 自动拟合指标已保存。
- [ ] home、极限、折叠和翻转姿态已可视化。
- [ ] 真机点云没有显示明显漏覆盖。
- [ ] 安全余量依据实际误差设置。
- [ ] self-collision ignore 已人工复核。
- [ ] 已完成 planning-only 测试。
- [ ] 已完成空场景低速测试。
- [ ] 已完成已知障碍物间距验证。
- [ ] 真机独立安全功能已启用。

## 15. 参考资料

- cuRobo 配置新机器人：
  <https://curobo.org/tutorials/1_robot_configuration.html>
- cuRobo Collision World Representation：
  <https://curobo.org/get_started/2c_world_collision.html>
- cuRobo Sphere Fit API：
  <https://curobo.org/_api/curobo.geom.sphere_fit.html>
- 本仓库 cuRobo sphere fitting 文档：
  `maniskill_curobo/external/curobo/docs/reference/sphere_fitting.rst`
- 本仓库 cuRobo self-collision 文档：
  `maniskill_curobo/external/curobo/docs/reference/self_collision.rst`
- 本仓库 RobotBuilder 示例：
  `maniskill_curobo/external/curobo/curobo/examples/getting_started/build_robot_model.py`
