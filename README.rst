Topaz plugin
============

This plugin allows to use Topaz programs within the Scipion framework.
It will allow to denoise, pre-process micrographs and pick particles within
Scipion.

`Topaz`_ is a pipeline for particle detection in cryo-electron
microscopy images which uses convolutional neural networks trained from
positive and unlabeled examples.

Setup
=====

For Users
---------

Install `Scipion3`_, follow the 'Topaz integration' instructions below and `install`_ the Topaz plugin.

For developers
--------------

1. For testing and develop this plugin, you need to use the Scipion v3.0.
   For that, just install Scipion from `GitHub`_, using the ‘devel’ branch.
2. Follow the 'Topaz integration' instructions below.
3. Clone this repository in you system: 
   ::

      cd
      git clone https://github.com/scipion-em/scipion-em-topaz
   
4. Install the Topaz plugin in devel mode:
   ::
      
      scipion installp -p ~/scipion-em-topaz --devel


Topaz integration
-----------------

The following steps assume that you have Anaconda or Miniconda installed on your computer.

| In ``~/.config/scipion/scipion.conf``: 
| Set CONDA_ACTIVATION_CMD variable in the Packages section.
| For example: ``CONDA_ACTIVATION_CMD = . ~/anaconda2/etc/profile.d/conda.sh``
| Notice the command starts with a period! This will source the conda.sh script.
  This is needed to activate the conda environment.
| For further information please visit the following website:
| https://github.com/conda/conda/blob/master/CHANGELOG.md#440-2017-12-20
| Set TOPAZ_ENV_ACTIVATION variable in the Packages section.
| For example: ``TOPAZ_ENV_ACTIVATION = conda activate topaz-0.2.3``
| This will activate the conda environment with the default name.


.. _Topaz: https://github.com/tbepler/topaz

.. _Scipion3: https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html

.. _install: https://scipion-em.github.io/docs/docs/scipion-modes/install-from-sources#step-4-installing-xmipp3-and-other-em-plugins

.. _GitHub: https://scipion-em.github.io/docs/docs/scipion-modes/install-from-sources#from-github
