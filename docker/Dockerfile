FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=8.0+PTX \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
      ninja-build && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/zerograsp
COPY . /opt/zerograsp

RUN grep -Ev '^(dwconv|ocnn) @ ' requirements.upstream.txt > /tmp/requirements.upstream.filtered.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.2.0 \
      torchvision==0.17.0 && \
    python -m pip install \
      torch-scatter \
      -f https://data.pyg.org/whl/torch-2.2.0+cu121.html && \
    python -m pip install \
      torch-cluster \
      -f https://data.pyg.org/whl/torch-2.2.0+cu121.html && \
    python -m pip install --no-build-isolation \
      "dwconv @ git+https://github.com/octree-nn/dwconv.git@ae53057eaf36dab01aa2727fcc93a749fd995af5#egg=dwconv" \
      "ocnn @ git+https://github.com/octree-nn/ocnn-pytorch.git" && \
    python -m pip install --no-build-isolation -r /tmp/requirements.upstream.filtered.txt && \
    python -m pip install graspnetAPI --no-deps && \
    python -m pip install transforms3d==0.4.2 autolab_core cvxopt grasp_nms && \
    cd third_party/octree_feature_extractor && \
    python setup.py install

ENV PYTHONPATH=/opt/zerograsp

ENTRYPOINT ["python", "/opt/zerograsp/run_mainline.py"]
