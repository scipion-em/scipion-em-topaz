# **************************************************************************
# *
# * Authors:     Daniel Del Hoyo Gomez (daniel.delhoyo.gomez@alumnos.upm.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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

import os, re
import time

import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pwem.protocols import ProtParticlePickingAuto

from topaz import convert, Plugin
from topaz.protocols.protocol_base import ProtTopazBase
from topaz.convert import (readSetOfCoordinates)

TOPAZ_COORDINATES_FILE = 'topaz_coordinates_file'
PICKING_DENOISE_FOLDER = 'picking_denoise_folder'
PICKING_PRE_FOLDER = 'picking_pre_folder'
PICKING_FOLDER = 'picking_folder'
MODEL_FOLDER = 'model_folder'

MICRO_BASE_FOLDER = "micrographs%(min)s-%(max)s"


class TopazProtPicking(ProtParticlePickingAuto, ProtTopazBase):
  """ Perform a picking using a topaz model """
  _label = 'picking'

  ADD_MODEL_TRAIN_TYPES = ["TopazTrained", "TopazGeneral"]
  ADD_MODEL_PRETRAINED = 0
  ADD_MODEL_GENERAL = 1

  GENERAL_MODELS = ["resnet16_u64", "resnet16_u32", "resnet8_u64", "resnet8_u32"]
  MODEL_RESNET16_U64 = 0
  MODEL_RESNET16_U32 = 1
  MODEL_RESNET8_U64 = 2
  MODEL_RESNET8_U32 = 3

  def __init__(self, **args):
    ProtParticlePickingAuto.__init__(self, **args)
    self.stepsExecutionMode = cons.STEPS_PARALLEL

  # -------------------------- DEFINE param functions -----------------------
  def _defineParams(self, form):
    ProtParticlePickingAuto._defineParams(self, form)
    form.addParam('modelInitialization', params.EnumParam,
                  choices=self.ADD_MODEL_TRAIN_TYPES,
                  default=self.ADD_MODEL_PRETRAINED,
                  label='Select model type',
                  help='If you set to *%s*, a topaz model object, '
                       'within this project, will be employed. If you set to *%s* a '
                       'pretrained model from topaz software will be used'
                       % tuple(self.ADD_MODEL_TRAIN_TYPES))

    form.addParam('prevTopazModel', params.PointerParam,
                  pointerClass='TopazModel',
                  condition='modelInitialization== %s' % self.ADD_MODEL_PRETRAINED, allowsNull=True,
                  label='Select topaz model',
                  help='Select a topaz model to continue from.')
    form.addParam('generalModel', params.EnumParam,
                  choices=self.GENERAL_MODELS, default=self.MODEL_RESNET16_U64,
                  condition='modelInitialization== %s' % self.ADD_MODEL_GENERAL,
                  label='Topaz general model',
                  help='A topaz NN model pretrained and provided in topaz sofware.'
                       '\nMight not be optimized for specific particles')

    form.addSection('Picking')
    form.addParam('radius', params.IntParam, default=8,
                  label='Particle radius (px)',
                  allowsPointers=True,
                  help='Pixel radius around particle centers to consider.')
    form.addParam('boxSize', params.IntParam, default=-1, expertLevel=cons.LEVEL_ADVANCED, allowsPointers=True,
                  label='Box size (px)', help='Box size in pixels. By default(-1): radius*2*scale')
    form.addParam('threshold', params.FloatParam, default=-6.0,
                  label='Extraction threshold',
                  help='log-likelihood score threshold at which to terminate region extraction. '
                       '\nValue -6 is p>=0.0025 (default: -6)'
                       '\nHigher values will mean a more restrictive picking')

    form.addParallelSection(threads=1, mpi=1)
    self._definePreprocessParams(form)
    self._defineStreamingParams(form)
    form.getParam('streamingBatchSize').setDefault(32)

  # -------------------------- INSERT steps functions -----------------------
  def _insertInitialSteps(self):
    self._defineFileDict()
    return []

  def _defineFileDict(self):
    """ Centralize how files are called for iterations and references. """
    pickingFolder = self._getTmpPath(MICRO_BASE_FOLDER)
    pickingDenoiseFolder = os.path.join(pickingFolder, "denoise")
    pickingPreFolder = os.path.join(pickingFolder, "preprocess")
    myDict = {
      MODEL_FOLDER: self._getExtraPath("model"),
      PICKING_FOLDER: pickingFolder,
      PICKING_DENOISE_FOLDER: pickingDenoiseFolder,
      PICKING_PRE_FOLDER: pickingPreFolder,
      TOPAZ_COORDINATES_FILE: os.path.join(pickingPreFolder,
                                           "topaz_coordinates%(min)s-%(max)s.txt")
    }

    self._updateFilenamesDict(myDict)

  # --------------------------- STEPS functions ------------------------------
  def _pickMicrograph(self, micrograph, *args):
    """Picking the given micrograph. """
    self._pickMicrographList([micrograph], *args)

  def _pickMicrographList(self, micList, *args):
    # Link or convert the whole set of micrographs to "batch" folders
    if len(micList) > 0:
        workingDir = self.getPickingFileName(micList, PICKING_FOLDER)
        pwutils.makePath(workingDir)

        convert.convertMicrographs(micList, workingDir)

        if self.doDenoise:
          denoisedDir = self.getPickingFileName(micList, PICKING_DENOISE_FOLDER)
          pwutils.makePath(denoisedDir)
          # denoise the micrographs in the batch folder, output in denoisedDir
          args = self.getDenoiseArgs(workingDir, denoisedDir)
          Plugin.runTopaz(self, 'topaz denoise', args)
          workingDir = denoisedDir

        # create preprocessed folder under the workingDir.
        # Now in the extra folder should be replaced in tmp folder
        preprocessedDir = self.getPickingFileName(micList, PICKING_PRE_FOLDER)
        pwutils.makePath(preprocessedDir)

        # preprocess the micrographs in the batch folder, output in preprocessedDir
        args = self.getPreprocessArgs(workingDir, preprocessedDir)
        Plugin.runTopaz(self, 'topaz preprocess', args)

        # perform prediction on the preprocessed micrographs
        if self.modelInitialization.get() == self.ADD_MODEL_PRETRAINED:
          modelFn = self.prevTopazModel.get().getPath()
        elif self.modelInitialization.get() == self.ADD_MODEL_GENERAL:
          modelFn = self.getEnumText('generalModel')

        # Launch process called extract which is rather a prediction
        args = ' -t {}'.format(self.threshold.get())
        args += ' -r %d' % self.radius.get()
        args += ' -m %s' % modelFn
        args += ' -o %s' % self.getPickingFileName(micList,
                                                   TOPAZ_COORDINATES_FILE)
        args += ' --num-workers %d' % self.numberOfThreads
        args += ' --device %(GPU)s'  # Add GPU that will be set by the executor
        args += ' %s/*.mrc' % preprocessedDir

        Plugin.runTopaz(self, 'topaz extract', args)

  def readCoordsFromMics(self, outputDir, micDoneList, outputCoords):
    """ Read the coordinates from a given list of micrographs """

    scale = self.scale.get()

    minMaxs = self.getPickingMinMax(micDoneList)
    for kMin, kMax in minMaxs:
        pickingFileName = self._getFileName(TOPAZ_COORDINATES_FILE, **{"min": kMin, 'max': kMax})


        readSetOfCoordinates(pickingFileName, outputCoords.getMicrographs(),
                             outputCoords, scale)

    if self.boxSize.get() == -1:
      boxSize = self.radius.get() * 2 * scale
    else:
      boxSize = self.boxSize.get()
    outputCoords.setBoxSize(boxSize)

  # --------------------------- UTILS functions --------------------------
  def getPickingFileName(self, micList, key):
    return self._getFileName(key, **{"min": micList[0].strId(),
                                     'max': micList[-1].strId()})

  def getPickingMinMax(self, micList):
      '''From the list of done micrographs, recover the corresponding picking filenames, which can result to be
       in several files due to GPU parallelization'''
      minId, maxId = micList[0].strId(), micList[-1].strId()

      regexPattern = re.sub(r"%\((\w+)\)s", r"(?P<\1>.+)", os.path.basename(MICRO_BASE_FOLDER))
      regex = re.compile(f"^{regexPattern}$")

      matches = []
      for name in os.listdir(self._getTmpPath()):
          m = regex.match(name)
          if m:
              kMin, kMax = m.groupdict().values()
              if int(kMin) >= int(minId) and int(kMax) <= int(maxId):
                  matches.append((kMin, kMax))

      return matches



  def _validate(self):
    validateMsgs = []
    if self.modelInitialization.get() == self.ADD_MODEL_PRETRAINED:
      if self.prevTopazModel.get() is None:
        validateMsgs.append('Model not ready')

    nGPUs = len(getattr(self, params.GPU_LIST).get().split())
    if self.numberOfThreads.get() <= nGPUs and nGPUs != 1:
        validateMsgs.append('The number of threads must be at least 1 more than the number of assigned GPUs, since '
                            'this software can only run 1 GPU per thread (and 1 thread is reserved for Scipion main)')
    return validateMsgs