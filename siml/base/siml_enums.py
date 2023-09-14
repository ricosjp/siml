from enum import Enum


class SimlFileExtType(Enum):
    NPY = ".npy"
    NPYENC = ".npy.enc"
    NPZ = ".npz"
    NPZENC = ".npz.enc"
    PKL = ".pkl"
    PKLENC = ".pkl.enc"
    PTH = ".pth"
    PTHENC = ".pth.enc"
    YAML = ".yaml"
    YML = ".yml"
    YAMLENC = ".yaml.enc"
    YMLENC = ".yml.enc"


class ModelSelectionType(Enum):
    BEST = "best"
    LATEST = "latest"
    TRAIN_BEST = "train_best"
    SPECIFIED = "specified"
    DEPLOYED = "deployed"


class LossType(Enum):
    MSE = "mse"


class DirectoryType(Enum):
    RAW = 0
    INTERIM = 1
    PREPROCESSED = 2
