# FROM nvidia/cuda:10.1-base
FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends \
  git \
  make \
  gcc \
  curl \
  g++ \
  lsb-release wget software-properties-common \
  libedit-dev \
  python3.7 \
  python3.7-dev \
  python3.7-distutils \
  python3-pip \
  python-pydot python-pydot-ng graphviz \
  && bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" \
  && wget https://apt.llvm.org/llvm.sh \
  && chmod +x llvm.sh \
  && ./llvm.sh 10 \
  && ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

ENV LLVM_CONFIG /usr/bin/llvm-config-10


COPY ./pyproject.toml /src/pyproject.toml
# COPY ./poetry.lock /src/poetry.lock
WORKDIR /src

ARG ricos_pypi_key
ENV PATH $PATH:/root/.poetry/bin

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.7 \
  && sed -i.bak s/python/python3.7/g ~/.poetry/bin/poetry \
  && python3.7 -m pip install -U setuptools \
  && python3.7 -m pip install wheel \
  && poetry config virtualenvs.create false \
  && poetry config repositories.ricos https://pypi.ritc.jp \
  && poetry config http-basic.ricos ricos $ricos_pypi_key \
  && poetry install \
  && python3.7 -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html \
  && python3.7 -m pip install torch-scatter==2.0.5+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && python3.7 -m pip install torch-sparse==0.6.7+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && python3.7 -m pip install torch-cluster==1.5.7+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && python3.7 -m pip install torch-spline-conv==1.2.0+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html \
  && python3.7 -m pip install torch-geometric
