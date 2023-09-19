CI_REGISTRY_IMAGE ?= registry.ritc.jp/ricos/machine_learning/siml
GHCR_IMAGE ?= ghcr.io/ricosjp/siml
TAG ?= 0.2.9

all: build

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY_IMAGE)
endif

build: Dockerfile
	rm -r ./dist || true
	mkdir dist
	cp ../pyproject.toml ../README.md ./dist
	cp -r ../siml ./dist
	find ./dist/siml | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
	docker build -t $(CI_REGISTRY_IMAGE):$(TAG) \
		-f Dockerfile .

build_cpu:
	rm -r ./dist || true
	mkdir dist
	cp ../pyproject.toml ../README.md ./dist
	cp -r ../siml ./dist \
	&& find ./dist/siml | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
	docker build -t $(CI_REGISTRY_IMAGE)/cpu:$(TAG) \
		-f Dockerfile_cpu .

in: build
	docker run -it --gpus all --rm $(CI_REGISTRY_IMAGE):$(TAG) /bin/bash

push: login build build_cpu
	docker push $(CI_REGISTRY_IMAGE):$(TAG)
	docker push $(CI_REGISTRY_IMAGE)/cpu:$(TAG)

ghcr: ghcr_login
	docker pull $(CI_REGISTRY_IMAGE)/cpu:$(TAG)
	docker tag $(CI_REGISTRY_IMAGE)/cpu:$(TAG) $(GHCR_IMAGE)/cpu:$(TAG)
	docker push $(GHCR_IMAGE)/cpu:$(TAG)

ghcr_login:
	docker login $(GHCR_IMAGE)
