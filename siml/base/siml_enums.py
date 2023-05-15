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
    YAML = ".yml"
    YAMLENC = ".yml.enc"


class ModelSelectionType(Enum):
    BEST = 0
    LATEST = 1
    TRAIN_BEST = 2
    SPECIFIED = 3
    DEPLOYED = 4


class LossType(Enum):
    MSE = "mse"


class DirectoryType(Enum):
    RAW = 0
    INTERIM = 1
    PREPROCESSED = 2
