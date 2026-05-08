# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *              Peter Horvath (phorvath@cnb.csic.es) [2]
# *
# * [1] SciLifeLab, Stockholm University
# * [2] I2PC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
from pwem.protocols import EMProtocol
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons

class ProtTopazBase(EMProtocol):
    """
    Base class for Topaz protocols providing preprocessing and denoising
    utilities for micrograph preparation.

    AI Generated:

    Topaz Base Preprocessing (ProtTopazBase) — User Manual
        Overview

        The ProtTopazBase class provides the common preprocessing
        operations used by Topaz-based workflows.

        Its main purpose is to prepare cryo-EM micrographs before
        downstream particle picking, training, or model inference.

        In practical cryo-EM workflows, raw micrographs often benefit
        from denoising and downsampling before automated analysis.
        These preprocessing steps improve computational efficiency and
        may also improve the robustness of downstream particle
        detection.

        From a methodological perspective, this protocol does not
        perform particle picking itself. Instead, it defines the common
        preprocessing framework used by higher-level Topaz protocols.

        Preprocessing Workflow

        The preprocessing framework supports two optional stages:

            1. Denoising
            2. Downsampling and preprocessing

        These stages can be used independently or sequentially,
        depending on the downstream analysis requirements.

        Denoising

        Denoising is optional and can be enabled by the user.

        When enabled, each micrograph is processed using one of the
        available Topaz denoising neural network models.

        Available models include:

            - unet
            - unet-small
            - fcnn
            - affineresnet8

        The choice of model determines the architecture used to reduce
        high-frequency noise while preserving structural signal.

        In practical cryo-EM processing, denoising can improve particle
        visibility, especially in low signal-to-noise datasets.

        Patch-Based Processing

        Large micrographs may exceed available GPU memory.

        To address this, the protocol optionally supports patch-based
        denoising.

        When a patch size is provided, each micrograph is processed in
        smaller independent image tiles.

        This is particularly useful for high-resolution micrographs or
        systems with limited GPU memory.

        If no patch size is provided, denoising is applied to the whole
        micrograph in a single pass.

        Denoising Normalization

        If no advanced denoising options are specified, the protocol
        automatically adds normalization during denoising.

        This normalization step helps standardize image intensity
        values, which can improve the stability of downstream neural
        network inference.

        Downsampling and Preprocessing

        The preprocessing stage performs image downsampling.

        The user specifies a scale factor that controls the reduction in
        image size.

        In typical Topaz workflows, micrographs are downsampled so that
        the resulting pixel size is approximately 8 Å.

        This substantially reduces computational cost while preserving
        enough structural information for particle detection.

        From a practical perspective, downsampling is especially useful
        during large-scale automated particle picking or model training.

        GPU and Parallel Execution

        Both denoising and preprocessing support GPU execution.

        The protocol includes a hidden GPU selection parameter so that
        Topaz jobs can be assigned to specific GPU devices.

        It also supports multithreaded CPU preprocessing through the
        number of worker threads.

        This combination allows efficient processing of large cryo-EM
        datasets.

        Advanced Command-Line Options

        Both denoising and preprocessing expose advanced optional
        command-line parameters.

        These options allow expert users to pass custom Topaz
        arguments directly to the underlying execution commands.

        In routine workflows, the default settings are usually
        sufficient, but advanced users may use these parameters to
        optimize performance for unusual datasets or specialized
        hardware environments.

        Model Management Utilities

        The class also provides utility functions for model handling.

        In particular, it can automatically locate the last trained
        model in a models directory.

        This is useful in training workflows where multiple checkpoint
        files may exist and the most recent epoch must be selected
        automatically.

        Practical Recommendations

        In most cryo-EM workflows, denoising is especially useful when
        working with low-contrast micrographs or noisy acquisition
        conditions.

        Downsampling is generally recommended before large-scale
        particle picking because it greatly improves speed while
        usually preserving sufficient particle information.

        Patch-based denoising should be used whenever GPU memory
        becomes a limiting factor.

        Final Perspective

        For cryo-EM users, ProtTopazBase provides the foundational
        preprocessing layer for Topaz-based analysis.

        Although it does not directly produce biological results, its
        impact is highly practical: good preprocessing often improves
        the quality, speed, and robustness of all downstream particle
        detection workflows.
    """
  def __init__(self, **args):
    EMProtocol.__init__(self, **args)

  def _definePreprocessParams(self, form):
    form.addSection('Pre-process')
    group = form.addGroup('Denoise')
    group.addParam('doDenoise', params.BooleanParam, default=False,
                   label="Denoise micrographs?")
    group.addParam('modelDenoise', params.EnumParam, default=0,
                   condition='doDenoise',
                   choices=['unet', 'unet-small', 'fcnn', 'affineresnet8'],
                   label='Model',
                   help='Denoising model to use on micrographs.')
    group.addParam('patchSize', params.IntParam, default=-1,
                   label='Patch Size', condition='doDenoise',
                   help='Process each micrograph in patches of this size.\n'
                        'This is useful when using GPU processing and the micrographs '
                        'are too large to be denoised in one shot on your GPU. '
                        'By default (<0), it is not used')
    group.addParam('denoiseExtra', params.StringParam, default='',
                   expertLevel=cons.LEVEL_ADVANCED, condition='doDenoise',
                   label="Advanced options",
                   help="Provide advanced command line options here.")

    group = form.addGroup('Pre-process')
    group.addParam('scale', params.IntParam, default=4,
                   label='Scale factor',
                   help='Scaling factor for image downsampling.\n'
                        'Downsample such that the resulting pixel size '
                        'is about 8 Angstroms.')
    group.addParam('preExtra', params.StringParam, default='',
                   expertLevel=cons.LEVEL_ADVANCED,
                   label="Advanced options",
                   help="Provide advanced command line options here.")

    form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                   expertLevel=cons.LEVEL_ADVANCED,
                   label="Choose GPU IDs",
                   help="GPU may have several cores. Set it to zero"
                        " if you do not know what we are talking about."
                        " First core index is 0, second 1 and so on.")


  #UTILS for preprocess steps
  def getDenoiseArgs(self, inputDir, outDir):
      args = ' %s/*.mrc -o %s/' % (inputDir, outDir)
      args += ' --model %s' % self.getEnumText('modelDenoise')
      args += ' --device %(GPU)s'  # Add GPU that will be set by the executor
      if self.patchSize.get() > 0:
        args += ' --patch-size %s' % self.patchSize.get()

      if self.denoiseExtra.hasValue():
        args += ' ' + self.denoiseExtra.get()
      else:
        args += ' --normalize'

      return args

  def getPreprocessArgs(self, inputDir, outDir):
    args = " %s/*.mrc -o %s/" % (inputDir, outDir)
    args += " --scale %d " % self.scale.get()
    args += ' --num-workers %d' % self.numberOfThreads
    args += ' --device %(GPU)s'  # Add GPU that will be set by the executor

    if self.preExtra.hasValue():
      args += ' ' + self.preExtra.get()

    return args

  def getOutputModelPath(self):
    return self.MODEL

  def getLastEpochModel(self, modelsDir, ext='.sav'):
    '''Return the last trained model, in alphabetic order (last trained epoch) in modelsDir'''
    modelFn = 'model.sav'
    for file in os.listdir(modelsDir):
      if ext in file:
        modelFn = file
    return os.path.join(modelsDir, modelFn)





