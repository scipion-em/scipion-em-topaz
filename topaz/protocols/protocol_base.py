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
  '''Base for topaz protocols including preprocessing parameters and methods'''
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





