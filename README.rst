============
Topaz plugin
============

This plugin allows to use Topaz programs within the Scipion framework.
It can denoise, pre-process micrographs and pick particles within Scipion.

`Topaz <https://github.com/tbepler/topaz>`_ is a pipeline for particle detection in cryo-electron
microscopy images which uses convolutional neural networks trained from
positive and unlabeled examples.

.. image:: https://img.shields.io/pypi/v/scipion-em-topaz.svg
        :target: https://pypi.python.org/pypi/scipion-em-topaz
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/l/scipion-em-topaz.svg
        :target: https://pypi.python.org/pypi/scipion-em-topaz
        :alt: License

.. image:: https://img.shields.io/pypi/pyversions/scipion-em-topaz.svg
        :target: https://pypi.python.org/pypi/scipion-em-topaz
        :alt: Supported Python versions

.. image:: https://img.shields.io/sonar/quality_gate/scipion-em_scipion-em-topaz?server=https%3A%2F%2Fsonarcloud.io
        :target: https://sonarcloud.io/dashboard?id=scipion-em_scipion-em-topaz
        :alt: SonarCloud quality gate

.. image:: https://img.shields.io/pypi/dm/scipion-em-topaz
        :target: https://pypi.python.org/pypi/scipion-em-topaz
        :alt: Downloads

Installation
------------

You will need to use 3.0+ version of Scipion to be able to run these protocols. To install the plugin, you have two options:

a) Stable version

    It can be installed in user mode via Scipion plugin manager (**Configuration** > **Plugins**) or using the command line:

    .. code-block::

        scipion installp -p scipion-em-topaz

b) Developer's version

    * download repository

    .. code-block::

        git clone -b devel https://github.com/scipion-em/scipion-em-topaz.git

    * install

    .. code-block::

        scipion installp -p /path/to/scipion-em-topaz --devel

Topaz software will be installed automatically with the plugin but you can also use an existing installation by providing *TOPAZ_ENV_ACTIVATION* (see below).

**Important:** you need to have conda (miniconda3 or anaconda3) pre-installed to use this program.

To check the installation you can run the plugin's tests:

``scipion tests topaz.tests.test_protocol_topaz.TestTopaz``


Configuration variables
-----------------------

**CONDA_ACTIVATION_CMD**: If undefined, it will rely on conda command being in the
PATH (not recommended), which can lead to execution problems mixing scipion
python with conda ones. One example of this could can be seen below but
depending on your conda version and shell you will need something different:

CONDA_ACTIVATION_CMD = eval "$(/extra/miniconda3/bin/conda shell.bash hook)"

**TOPAZ_ENV_ACTIVATION** (default = conda activate topaz-0.2.5):
Command to activate the Topaz environment.

Supported versions
------------------

0.2.3, 0.2.4, 0.2.5

Protocols
---------

    * import training model
    * picking
    * training

References
----------

    * Bepler, T. et al. Positive-unlabeled convolutional neural networks for particle picking in cryo-electron micrographs. arXiv (2018).
