FROM tiangolo/python-machine-learning:cuda9.1-python3.7

RUN apt-get update \
 && apt-get -y install cuda-libraries-9-1 cuda-nvcc-9-1 cuda-libraries-dev-9-1 \
 && python3 -m pip install 'cupy-cuda91==7.0.0b2' \
 && apt-get clean      \
 && rm -rf /var/lib/apt/lists/*

