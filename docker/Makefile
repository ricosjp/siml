CI_REGISTRY_IMAGE  ?= registry.ritc.jp/ricos/machine_learning/siml

all: build

login:
ifeq ($(CI_BUILD_TOKEN),)
	docker login $(CI_REGISTRY_IMAGE)
else
	docker login -u gitlab-ci-token -p $(CI_BUILD_TOKEN) $(CI_REGISTRY_IMAGE)
endif

build:
	docker build -t $(CI_REGISTRY_IMAGE) . 


push: login build
	docker push $(CI_REGISTRY_IMAGE)

