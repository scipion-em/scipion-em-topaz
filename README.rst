Topaz plugin
============

This plugin allows to use Topaz programs within the Scipion framework.
It will allow to pre-process micrographs and pick particles within
Scipion.

`Topaz`_ is a pipeline for particle detection in cryo-electron
microscopy images using convolutional neural networks trained from
positive and unlabeled examples.

Setup
=====
Scipion installation
--------------------

For testing this plugin, we need to use the “pluginization version” of
Scipion. For that, just install Scipion from GitHub, using the
‘devel-pluginization’ branch. Follow this instructions:
https://github.com/I2PC/scipion/wiki/How-to-Install-Scipion-from-sources#from-github

Topaz integration
-----------------

| STEP1:
| In ~/.config/scipion/scipion.conf: Set CONDA_ACTIVATION_CMD variable
  in the Packages section. For example: CONDA_ACTIVATION_CMD = .
  ~/anaconda2/etc/profile.d/conda.sh .This will source the conda.sh script.
  This is needed to activate the conda environment. For further information please
  visit the following website:
  https://github.com/conda/conda/blob/master/CHANGELOG.md#440-2017-12-20
  Set TOPAZ_ACTIVATION_CMD variable in the Packages
  section. For example: TOPAZ_ACTIVATION_CMD = conda activate topaz This
  will activate the conda environment with the default name topaz.

| STEP2:
| Type ./scipion installp -p topaz or you can also install it from the
  plugin manager (1, launch scipion; 2, go to configuration; 3, select
  plugins). This will conda install topaz for you.

.. _Topaz: https://github.com/tbepler/topaz