# ZeroGrasp Mainline Minimal Package

这个精简包只保留了 **ZeroGrasp 主干线推理** 所需的最小文件集合，用来完成：

1. 读取一组 `rgb + depth + mask + camera_info`
2. 运行 ZeroGrasp 主干线
3. 输出三维重建点云 `*.ply`
4. 输出抓取候选 `*.grasp.npy`
5. 额外整理一份人可读的 `recommended_grasp_top1.json`

## 保留内容

- `demo.py`
- `main.py`
- `configs/demo.yaml`
- `zerograsp/` 主模型代码
- `ofe/` 已编译的 octree feature extractor 运行时包
- `checkpoints/zerograsp_cvpr2025/zerograsp_demo.ckpt`
- `run_mainline.py` 主线运行器
- `examples/demo_input/000.*` 单个示例输入
- `requirements.upstream.txt` upstream 依赖参考

## 不再保留

- Isaac Sim 相关脚本
- 预实验统计、画图、执行验证脚本
- 大量历史 outputs / reports / snapshots
- 训练、批处理、数据集适配等非主干文件
- 多余 demo 样本

## 已验证环境

当前包是在下面这个 Python 环境中验证的：

- `/home/zyc/miniconda3/envs/graduate/bin/python`

如果你沿用这套环境，最稳。

## 运行方式

在包根目录执行：

```bash
/home/zyc/miniconda3/envs/graduate/bin/python run_mainline.py \
  --img_path examples/demo_input/000.rgb.png \
  --depth_path examples/demo_input/000.depth.png \
  --mask_path examples/demo_input/000.mask.png \
  --camera_info_path examples/demo_input/000.meta.yml \
  --output_dir example_run
```

## Docker 运行方式

这个精简包现在也带了 Docker 主干线运行支持，但要注意：

- 需要目标机器已经安装 **Docker + NVIDIA Container Toolkit**
- 当前工作机上没有 Docker，所以这里只能交付构建文件，不能本地完成镜像实测

在包根目录执行：

```bash
./docker/build_image.sh
```

然后把外部输入喂进去：

```bash
./docker/run_mainline_in_docker.sh \
  /path/to/input.rgb.png \
  /path/to/input.depth.png \
  /path/to/input.mask.png \
  /path/to/input.meta.yml \
  /path/to/output_dir
```

容器里同样会输出重建点云、抓取候选和 `recommended_grasp_top1.json`。

## Docker 完整 ManiSkill + ZeroGrasp 流程

如果要把当前项目搬到另一台 **RTX 3090** 机器上跑完整流程，使用 full 镜像：

```bash
./docker/build_full_image.sh
```

构建出的默认镜像名是：

```bash
zerograsp-maniskill:3090
```

目标机器需要先安装：

- NVIDIA driver
- Docker
- NVIDIA Container Toolkit

运行 PickSingleYCB：

```bash
ZERO_GRASP_OUTPUT_ROOT=/path/to/runs \
./docker/run_full_pipeline_in_docker.sh \
  --run-name picksingle_seed42 \
  --env-id PickSingleYCB-v1 \
  --seed 42
```

运行 PickClutterYCB：

```bash
ZERO_GRASP_OUTPUT_ROOT=/path/to/runs \
./docker/run_full_pipeline_in_docker.sh \
  --run-name pickclutter_seed42 \
  --env-id PickClutterYCB-v1 \
  --seed 42
```

默认输入 mask 使用 `--mask-mode task-target`，也就是只把 ManiSkill 任务目标物体传给 ZeroGrasp，避免桌子、地面、机械臂或非目标物体进入抓取候选。想让 ZeroGrasp 在所有可见物体里自由选择时，改成 `--mask-mode all-objects`；旧的像素面积过滤方式保留为 `--mask-mode visible-area` 方便对比。

容器会直接调用：

```bash
python -m maniskill_codex.run_full_pipeline --no-conda
```

每次 run 的所有产物都会放在同一个文件夹下，包括：

- `zg_input/`：给 ZeroGrasp 的 `rgb.png`、`depth.png`、`mask.png`、`camera.json`
- `zg_output/recommended_grasp_top1.json`
- `grasp_projection.png`
- `execution.mp4`
- `logs/`
- `run_manifest.json`

如果要使用宿主机上已有的输入目录，可以把路径放在当前工作目录下，然后传容器内路径：

```bash
./docker/run_full_pipeline_in_docker.sh \
  --run-name existing_input_test \
  --input-dir /workspace/host/path/to/zg_input
```

默认 full 镜像为 3090 设置了 `TORCH_CUDA_ARCH_LIST=8.6`，并把生成的 `maniskill_codex/runs`、视频和历史 ZeroGrasp 输出排除在 Docker build context 外，避免镜像被实验产物撑大。

## 输出说明

运行结束后，`example_run/` 或你指定的输出目录下会有：

- `raw_outputs/*.ply`
  每个对象一个重建点云
- `raw_outputs/*.grasp.npy`
  每个对象一个抓取候选数组
- `recommended_grasp_top1.json`
  全局最高分推荐抓取的摘要
- `reconstruction_summary.json`
  重建结果文件索引
- `run_report.json`
  本次运行的整体记录

## 推荐抓取位姿说明

`recommended_grasp_top1.json` 里的位姿目前是 **相机坐标系** 下的推荐抓取位姿，包含：

- 抓取中心位置 `translation_m_camera`
- 抓取朝向矩阵 `rotation_matrix_camera`
- 分数、宽度、深度等信息

如果你后续有相机外参，再把它转换到世界坐标系即可。
