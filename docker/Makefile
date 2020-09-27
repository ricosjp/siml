CI_REGISTRY_IMAGE  ?= registry.ritc.jp/ricos/machine_learning/siml
TAG ?= 0.2.4

all: build

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY_IMAGE)
endif

build: Dockerfile
	cp ../pyproject.toml .
	docker build -t $(CI_REGISTRY_IMAGE):$(TAG) \
		-f Dockerfile --build-arg ricos_pypi_key=$(key) .

in: build
	docker run -it --gpus all --rm $(CI_REGISTRY_IMAGE):$(TAG) /bin/bash

push: login build
	docker push $(CI_REGISTRY_IMAGE):$(TAG)
