FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
  git \
  make \
  gcc \
  ssh \
  curl \
  libglu1-mesa-dev \
  libxrender1 \
  g++ \
  lsb-release wget software-properties-common \
  libedit-dev \
  libmetis-dev \
  && apt-get install -y --no-install-recommends \
  tzdata \
  python3.9 \
  python3.9-dev \
  python3.9-distutils \
  python3-pip \
  && apt-get install -y --no-install-recommends \
  python3-pydot python3-pydot-ng graphviz \
  && bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" \
  && wget https://apt.llvm.org/llvm.sh \
  && chmod +x llvm.sh \
  && ./llvm.sh 10 \
  && ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

ENV LLVM_CONFIG /usr/bin/llvm-config-10


COPY ./pyproject.toml /src/pyproject.toml
COPY ./README.md /src/README.md
COPY ./siml /src/siml
# COPY ./poetry.lock /src/poetry.lock
WORKDIR /src

ENV PATH $PATH:/root/.local/bin

RUN curl -sSL https://install.python-poetry.org | python3.9 - \
  && sed -i.bak s/python3/python3.9/g ~/.local/bin/poetry \
  && python3.9 -m pip install -U pip \
  && python3.9 -m pip install -U setuptools \
  && python3.9 -m pip install -U wheel \
  && python3.9 -m pip install --ignore-installed -U PyYAML \
  && python3.9 -m pip install Cython \
  && poetry config virtualenvs.create false \
  && poetry install \
  && python3.9 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
  && python3.9 -m pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && python3.9 -m pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && python3.9 -m pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && python3.9 -m pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && python3.9 -m pip install torch-geometric==2.2.0
