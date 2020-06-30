FROM nvidia/cuda:10.1-base

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends \
  git \
  make \
  gcc \
  curl \
  python3.7 \
  python3.7-dev \
  python3.7-distutils \
  python3-pip \
  python-pydot python-pydot-ng graphviz

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
  && poetry install
