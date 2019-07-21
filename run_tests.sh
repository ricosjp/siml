#!/bin/bash

if [ -x "$(command -v nvcc )" ]
then
  python3 -m unittest discover tests
else
  CHAINER_TEST_GPU_LIMIT=0 python3 -m pytest tests -m='not cudnn'
fi
