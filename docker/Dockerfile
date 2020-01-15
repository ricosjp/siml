FROM nvidia/cuda:10.1-base

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends \
  git \
  make \
  python3.7 \
  python3.7-dev \
  python3.7-distutils \
  python3-pip

COPY ./pyproject.toml /src/pyproject.toml
COPY ./lib /src/lib
WORKDIR /src

RUN python3.7 -m pip install -U setuptools \
  && python3.7 -m pip install -U pip poetry \
  && python3.7 -m poetry config virtualenvs.create false \
  && python3.7 -m poetry install \
