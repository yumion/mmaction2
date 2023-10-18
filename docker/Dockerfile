ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMENGINE="0.9.0"
ARG MMCV="2.0.1"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"
ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y git wget curl ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# for excecuting as a user
RUN mkdir -p /.cache /.local /tmp \
    && chmod -R 777 /.cache /.local /tmp

# Install MIM
RUN pip install openmim
# Install MMCV
ARG MMENGINE
ARG MMCV
RUN ["/bin/bash", "-c", "mim install mmengine==${MMENGINE}"]
RUN ["/bin/bash", "-c", "mim install mmcv==${MMCV}"]

# Install MMAction2
RUN conda clean --all
RUN git clone -b main https://github.com/yumion/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN mkdir -p /mmaction2/data
RUN pip install cython --no-cache-dir
RUN pip install -r requirements/optional.txt
RUN pip install --no-cache-dir -e .
