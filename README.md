# Topaz plugin
This plugin allows to use Topaz programs within the Scipion framework.
It will allow to pre-process micrographs and pick particles within Scipion.

[Topaz](https://github.com/tbepler/topaz) is a pipeline for particle detection in cryo-electron microscopy images using convolutional neural networks trained from positive and unlabeled examples.

# Setup for current development

## Scipion installation 

### Installing Scipion from source

For testing this plugin, we need to use the "pluginization version" of Scipion. For that, just install Scipion from GitHub, using the 'devel-pluginization' branch. Follow this instructions:
https://github.com/I2PC/scipion/wiki/How-to-Install-Scipion-from-sources#from-github

and use the `git checkout devel-pluginization`

### Installing Xmipp
Xmipp still is required and need to be installed separated, setting the SCIPION_HOME before compilation to use the same Scipion Python and libraries. 

Instructions can be found here: https://github.com/I2PC/xmipp

### Installing all scipion-em-XXX plugins

All plugins are now standard Python modules that can be installed with pip, but they are still under development. So, I would rather recommend to clone from GitHub and make sure you have the PYTHONPATH pointing to them. 

One option could be using this repo: https://github.com/delarosatrevin/scipion-em-plugins that contains all plugins as submodules, and sourcing the https://github.com/delarosatrevin/scipion-em-plugins/blob/master/bash-plugins.sh file to modify the PYTHONPATH. 

This repo (scipion-em-topaz) is one of the submodules.

## Topaz integration

Topaz is suposed to be installed (e.g, via conda enviroment as stated here: https://github.com/tbepler/topaz#installation)
and the load command notified to the scipion plugin through the TOPAZ_CONDA_ENV variable. For example, in my system I set the following before launching Scipion:

`export TOPAZ_CONDA_ENV='. /home/josem/installs/anaconda2/etc/profile.d/conda.sh; conda activate topaz-env'`

### Topaz training protocol
The first draft of this protocol seems to be running and launching correctly topaz command.

* convertInputStep: takes input SetOfCoordinates and create:
   * image_list_train.txt
   * particles_train.txt
   * image_list_test.txt
   * particles_test.txt
   * convert input micrographs to mrc
   * run: topaz preprocess to create the tiff files
* trainingStep: based on the previous step:
   * run: topaz train with the user supplied parameters 
* extractStep:
   * run: topaz extract to produce predicted_particles_all.txt
* createOutputStep:
   * read coordinates from .txt file and generated the output SetOfCoordinates
   * This is still not working since there are some weird micrographs IDs (image_name) produced in the .txt file.
   
To test the protocol, launch:
`scipion test topaz.tests.test_protocols.TestTopaz`




