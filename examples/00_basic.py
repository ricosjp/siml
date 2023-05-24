"""
Basic Usage of SiML
===================

SiML facilitates machine learning processes, including preprocessing, learning,
and prediction.
We will cover the entire pipeline of a machine learning process using the
gradient dataset example.

"""

###############################################################################
# Import necessary modules including :mod:`siml`.
# `FEMIO <https://ricosjp.github.io/femio/>`_ is used to generate data.

import pathlib
import shutil

import femio
import numpy as np
import siml


###############################################################################
# Clean up old data if exists.
shutil.rmtree('00_basic_data/raw', ignore_errors=True)
shutil.rmtree('00_basic_data/interim', ignore_errors=True)
shutil.rmtree('00_basic_data/preprocessed', ignore_errors=True)
shutil.rmtree('00_basic_data/model', ignore_errors=True)
shutil.rmtree('00_basic_data/inferred', ignore_errors=True)


###############################################################################
# Data generation
# ---------------
# First, we define a function to generate data and call it
# to create the dataset.


def generate_data(output_directory):
    # Generate a simple mesh
    n_x_element = np.random.randint(5, 10)
    n_y_element = np.random.randint(5, 10)
    n_z_element = 1
    fem_data = femio.generate_brick(
        'hex',
        n_x_element=n_x_element,
        n_y_element=n_y_element,
        n_z_element=n_z_element,
        x_length=n_x_element,
        y_length=n_y_element,
        z_length=n_z_element)

    # Generate scalar field phi and the gradient field associated to it
    scale = 1 / 5
    nodes = np.copy(fem_data.nodes.data)
    nodes[:, -1] = 0.  # Make pseudo 2D
    shift = np.random.rand(1, 3) / scale
    shift[:, -1] = 0
    square_norm = .5 * np.linalg.norm(nodes - shift, axis=1)**2
    phi = np.cos(square_norm * scale)[:, None]
    grad = - np.sin(square_norm * scale)[:, None] * scale * (nodes - shift)

    # Write data
    fem_data.nodal_data.update_data(
        fem_data.nodes.ids, {'phi': phi, 'grad': grad},
        allow_overwrite=False)
    fem_data.write(
        'ucd', output_directory / 'mesh.inp')
    return


n_train_sample = 20
for i in range(n_train_sample):
    generate_data(pathlib.Path(f"00_basic_data/raw/train/{i}"))

n_validation_sample = 5
for i in range(n_validation_sample):
    generate_data(pathlib.Path(f"00_basic_data/raw/validation/{i}"))

n_test_data = 5
for i in range(n_validation_sample):
    generate_data(pathlib.Path(f"00_basic_data/raw/test/{i}"))


###############################################################################
# If the process finished successfully, the data should look as follows
# (visualization using `ParaView <https://www.paraview.org/>`_).
#
# .. image:: ../../examples/00_basic_fig/grad_train.png
#   :width: 400
#
# Here, we consider the task to predict the gradient field
# (arrows in the figure above) from the input of the scalar field
# (color map in the figure above).
#
# Data preprocessing
# ------------------
# Here, we extract features from the generated dataset.
# Data generation and feature extraction is something SiML does not manage
# because the library does not know what simulation to run and what features to
# extract.
# Therefore, users should write some code for these two parts,
# although SiML (and FEMIO) can support it.
#
# Now, define a call-back function to extract features from the dataset.
# The function takes two arguments,
# :code:`femio.FEMData` object representing a sample in the dataset and
# :code:`pathlib.Path` object representing an output directory.


def conversion_function(fem_data, raw_directory):

    node = fem_data.nodes.data

    phi = fem_data.nodal_data.get_attribute_data('phi')
    grad = fem_data.nodal_data.get_attribute_data('grad')[..., None]

    # Generate renormalized adjacency matrix based on Kipf and Welling 2016
    nodal_adj = fem_data.calculate_adjacency_matrix_node()
    nodal_nadj = siml.prepost.normalize_adjacency_matrix(nodal_adj)

    # Generate IsoAM based on Horie et al. 2020
    nodal_isoam_x, nodal_isoam_y, nodal_isoam_z = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1, moment_matrix=True)

    dict_data = {
        'node': node,
        'phi': phi,
        'grad': grad,
        'nodal_nadj': nodal_nadj,
        'nodal_isoam_x': nodal_isoam_x,
        'nodal_isoam_y': nodal_isoam_y,
        'nodal_isoam_z': nodal_isoam_z,
    }
    return dict_data


###############################################################################
# From here, SiML can manage most of the process.
# Please download
# `data.yml
# <https://github.com/ricosjp/siml/examples/00_basic_data/data.yml>`_
# file and place it in the :code:`00_basic_data` directory.
# SiML uses YAML files as setting files to control its behavior.
# Basically, each setting component can be omitted, and if so,
# the default setting will be adopted.
# The relevant contents of the YAML file are as follows.
#
# .. code-block:: yaml
#
#   data:  # Data directory setting
#     raw: 00_basic_data/raw   # Row data
#     interim: 00_basic_data/interim  # Extracted features
#     preprocessed: 00_basic_data/preprocessed  # Preprocessed data
#     inferred: 00_basic_data/inferred  # Predicted data
#   conversion:  # Feature extraction setting
#     file_type: 'ucd'  # File type to be read
#     required_file_names:  # Files to be regarded as data
#       - '*.inp'
#
# As can be seen, the structure of the directory follows that of the
# `Cookiecutter Data Science
# <https://drivendata.github.io/cookiecutter-data-science/>`_.
#
# Now, generate a :class:`~siml.prepost.RawConverter` object by feeding the
# YAML file and perform feature extraction.


settings_yaml = pathlib.Path('00_basic_data/data.yml')
raw_converter = siml.preprocessing.converter.RawConverter.read_settings(
    settings_yaml, conversion_function=conversion_function)
raw_converter.convert()


###############################################################################
# Next, perform preprocessing, e.g., scaling of the data.
# The relevant part of the YAML file is as follows.
#
# .. code-block:: yaml
#
#   preprocess:  # Data scaling setting
#     node: std_scale  # Standardization without subtraction of the mean
#     phi: standardize   # Standardization
#     grad: std_scale
#     nodal_nadj: identity  # No scaling
#     nodal_isoam_x: identity
#     nodal_isoam_y: identity
#     nodal_isoam_z: identity

preprocessor = siml.preprocessing.ScalingConverter.read_settings(settings_yaml)
preprocessor.fit_transform()

###############################################################################
# Training
# --------
# Then, we move on to the training.
# Please download
# `isogcn.yml
# <https://github.com/ricosjp/siml/examples/00_basic_data/isogcn.yml>`_
# file and place it in the :code:`00_basic_data` directory.
# In the YAML file, the setting for the trainer is written as follows.
#
# .. code-block:: yaml
#
#   trainer:
#     output_directory: 00_basic_data/model  # Output directory
#     inputs:  # Input data specification
#       - name: phi  # Input data name
#         dim: 1  # phi's dimention
#     support_input:  # Support inputs e.g. adjacency matrix
#       - nodal_isoam_x
#       - nodal_isoam_y
#       - nodal_isoam_z
#     outputs:
#       - name: grad  # Output data name
#         dim: 1  # gradient's dimention (the shape is in [n, 3, 1], so 1)
#     prune: false
#     n_epoch: 100  # The nmber of epochs
#     log_trigger_epoch: 1  # The period to log the training
#     stop_trigger_epoch: 5  # The period to condider early stopping
#     seed: 0  # The rondom seed
#     lazy: false  # If true, data is read lazily rather than on-memory
#     batch_size: 4  # The size of the batch
#     num_workers: 0  # The number of processes to load data (0 means serial)
#     figure_format: png  # Format of the output figures (the default is pdf)
#
# In the same file, the setting for the machine learning model is also written.
# In this example, we use `IsoGCN <https://arxiv.org/abs/2005.06316>`_
# (Horie et al. ICLR 2021).
# We can try many machine learning trials with various training and model
# settings by editing the YAML file.


isogcn_yaml = pathlib.Path('00_basic_data/isogcn.yml')
trainer = siml.trainer.Trainer(isogcn_yaml)
trainer.train()


###############################################################################
# The results of the training is stored in
# :code:`00_basic_data/model`. If you remove :code:`output_directory` line
# in the YAML file, the output directory will be determined automatically.
#
# ::
#
#   00_basic_data/model
#   ├── log.csv               # Logfile of the training
#   ├── network.png           # Network structure figure
#   ├── plot.png              # Loss-epoch plot
#   ├── settings.yml          # Trainin setting file for reproducibility
#   ├── snapshot_epoch_1.pth  # Model parameter at the epoch 1
#   ├── snapshot_epoch_2.pth
#   ├── snapshot_epoch_3.pth
#   .
#   .
#   .
#
# The network structure used in the training is shown below.
#
# .. image:: ../../examples/00_basic_data/model/network.png
#   :width: 200
#
# The loss vs. epoch curve is shown below.
#
# .. image:: ../../examples/00_basic_data/model/plot.png
#   :width: 400
#
# Prediction
# ----------
# Using the trained model, we can make a prediction.
# In the isogcn YAML file, the setting for inference is also written.

inferer = siml.inferer.Inferer.read_settings_file(
    isogcn_yaml, model_path=trainer.setting.trainer.output_directory)
inferer.infer(
    data_directories=[pathlib.Path('00_basic_data/preprocessed/test')],
)


###############################################################################
# The predicted data is stored in
# :code:`00_basic_data/inferred/model_[date]/test`
# (:code:`[date]` depends on the date when you run this script.)
#
# The structure of the directory is as follows.
#
# ::
#
#   00_basic_data/inferred/model_[date]
#    ├── log.csv           # Summary file
#    ├── settings.yml      # Setting used to prediction (for reproducibility)
#    └── test
#        ├── 0
#        │   ├── grad.npy  # Predicted gradient
#        │   ├── mesh.inp  # AVD UCD format file for visualization
#        │   └── phi.npy   # Input data
#        ├── 1
#        │   ├── grad.npy
#        │   ├── mesh.inp
#        │   └── phi.npy
#        .
#        .
#        .
#
# The predicted result will look as follows
# (left: ground truth, right: prediction). Looks good!
#
# .. image:: ../../examples/00_basic_fig/res.png
#   :width: 400
