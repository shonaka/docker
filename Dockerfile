# Sho's custom docker file for research
FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER Sho Nakagome <snakagome@uh.edu>

# Partially from https://github.com/uber/horovod/blob/master/Dockerfile
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=1.10.0
ENV PYTORCH_VERSION=0.4.0
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
ENV NCCL_VERSION=2.2.13-1+cuda9.0

# Setting Python version
ARG python=3.5
ENV PYTHON_VERSION=${python}
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        gcc \
        g++ \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# ===== Other external libraries for Python =====
# Install some python libraries
RUN pip install numpy scipy pandas scikit-learn matplotlib Cython hyperopt seaborn jupyter ruamel.yaml

# Install xgboost
RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    make -j4 && \
    cd python-package; python setup.py install

# Install lightgbm
RUN git clone --recursive https://github.com/Microsoft/LightGBM && \
    cd LightGBM/python-package && python setup.py install

# ===== Deep learning related Python libraries =====
# Install TensorFlow and Keras
RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py

# Install PyTorch
RUN PY=$(echo ${PYTHON_VERSION} | sed s/\\.//); \
    if echo ${PYTHON_VERSION} | grep ^3 >/dev/null; then \
        pip install http://download.pytorch.org/whl/cu90/torch-${PYTORCH_VERSION}-cp${PY}-cp${PY}m-linux_x86_64.whl; \
    else \
        pip install http://download.pytorch.org/whl/cu90/torch-${PYTORCH_VERSION}-cp${PY}-cp${PY}mu-linux_x86_64.whl; \
    fi; \
    pip install torchvision

# Install tflearn
RUN pip install tflearn

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar zxf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download examples
RUN apt-get install -y --no-install-recommends subversion && \
    svn checkout https://github.com/uber/horovod/trunk/examples && \
    rm -rf /examples/.svn

# ===== Some tweak to avoid import error when using this container in pycharm =====
ENV     CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu/
ENV     CUDA_TOOLKIT_PATH=/usr/local/cuda/

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
RUN mkdir -p /usr/local/nvidia/lib64/ \
              && cd /usr/local/nvidia/lib64/ \
              && ln -s /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so libcuda.so.1 \
              && ldconfig /usr/local/cuda/lib64

EXPOSE 8888 6006
