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
  && ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config \
  && rm -rf /var/lib/apt/lists/*

ENV LLVM_CONFIG /usr/bin/llvm-config-10


COPY ./dist/pyproject.toml /src/pyproject.toml
COPY ./dist/README.md /src/README.md
COPY ./dist/siml /src/siml
# COPY ./poetry.lock /src/poetry.lock
WORKDIR /src

# python path
ENV PYTHONPATH "$PYTHONPATH:/usr/lib/python3.9/site-packages"

# poerty
ENV PATH "$PATH:/root/.local/bin"

# Default version of python3 is python3.8.
# When virtual environment is not created by poetry,
# poetry use default version of python.
# In order to change default python version to python3.9,
# create symbolic link of python3.9
RUN curl -sSL https://install.python-poetry.org | python3.9 - \
  && unlink /usr/bin/python3 \
  && ln -s /usr/bin/python3.9 /usr/bin/python3 \
  && python3.9 -m pip install -U pip \
  && python3.9 -m pip install -U setuptools \
  && python3.9 -m pip install -U wheel \
  && python3.9 -m pip install --ignore-installed -U PyYAML \
  && python3.9 -m pip install Cython \
  && poetry config virtualenvs.create false \
  && poetry install \
  && poetry run python3 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
  && poetry run python3 -m pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && poetry run python3 -m pip install torch-sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && poetry run python3 -m pip install torch-cluster==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && poetry run python3 -m pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu111.html \
  && poetry run python3 -m pip install torch-geometric==2.2.0 \
  && poetry run python3 -m pip install -U "numpy<1.24"
