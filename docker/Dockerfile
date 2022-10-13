FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

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

# TODO: poetry not working due to mayavi, will investigate a solution
RUN \
  curl -sSL https://install.python-poetry.org | python3.9 - \
  && python3.9 -m pip install -U pip \
  && python3.9 -m pip install -U setuptools \
  && python3.9 -m pip install -U wheel \
  && python3.9 -m pip install --ignore-installed -U PyYAML \
  && python3.9 -m pip install Cython \
  && python3.9 -m pip install numpy \
  && python3.9 -m pip install vtk==9.1.0 \
  && python3.9 -m pip install mayavi==4.8 \
  && python3.9 -m pip install femio \
  && python3.9 -m pip install PyYAML einops joblib optuna pandas pycryptodomex pydot \
  && python3.9 -m pip install sqlalchemy PyQt5 toml metis \
  && python3.9 -m pip install chainer pytest autopep8 sphinx sphinx_rtd_theme sphinx-gallery flake8 \
  && python3.9 -m pip install torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 \
  && python3.9 -m pip install pytorch-ignite \
  && python3.9 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
  # && poetry config virtualenvs.create false \
  # && poetry install
