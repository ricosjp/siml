FROM nvidia/cuda:10.1-base

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends \
  git \
  make \
  gcc \
  python3.7 \
  python3.7-dev \
  python3.7-distutils \
  python3-pip

COPY ./pyproject.toml /src/pyproject.toml
WORKDIR /src

RUN python3.7 -m pip install -U setuptools \
  && python3.7 -m pip install wheel \
  && python3.7 -m pip install -U pip poetry \
  && python3.7 -m poetry config virtualenvs.create false \
  && python3.7 -m poetry install \
  && python3.7 -m poetry update
