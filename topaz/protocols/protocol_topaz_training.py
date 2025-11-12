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
import numpy as np

import pyworkflow as pw
import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pwem.protocols import ProtParticlePicking
from pwem.objects import SetOfMicrographs, SetOfCoordinates
from pwem.emlib.image import ImageHandler

from topaz.protocols.protocol_base import ProtTopazBase
from topaz import convert, Plugin
from topaz.convert import (CsvMicrographList, CsvCoordinateList, micId2MicName)
from topaz.objects import TopazModel


MODEL_FOLDER = 'model_folder'
TRAINING = 'training'
TRAINING_MIC = 'trainingMic'
TRAININGDENOISE = 'trainingdenoise'
TRAININGPREPROCESS = 'trainingpreprocess'
TRAININGPRE_MIC = 'trainingpreMic'
TRAININGLIST = 'traininglist'
TRAININGTEST = 'trainingtest'
PARTICLES_TEST_TXT = 'particles_test.txt'
PARTICLES_TRAIN_TXT = 'particles_train.txt'


class TopazProtTraining(ProtParticlePicking, ProtTopazBase):
  """ Train and save a Topaz model"""
  _label = 'training'

  ADD_MODEL_TRAIN_TYPES = ["New", "TopazModel"]
  ADD_MODEL_TRAIN_NEW = 0
  ADD_MODEL_TRAIN_MODEL = 1

  def __init__(self, **args):
    ProtParticlePicking.__init__(self, **args)
    self.stepsExecutionMode = cons.STEPS_PARALLEL

  # -------------------------- DEFINE param functions -----------------------
  def _defineParams(self, form):
    ProtParticlePicking._defineParams(self, form)
    form.addParam('inputCoordinates', params.PointerParam,
                  pointerClass='SetOfCoordinates',
                  label='Input coordinates', important=True,
                  help='Select the SetOfCoordinates to be used for '
                       'training.')

    form.addParam('modelInitialization', params.EnumParam,
                  choices=self.ADD_MODEL_TRAIN_TYPES,
                  default=self.ADD_MODEL_TRAIN_NEW,
                  label='Select model type',
                  help='If you set to *%s*, a new model randomly initialized will be '
                       'employed. If you set to *%s*, a model trained in a previous run, '
                       'within this project, will be employed.'
                       % tuple(self.ADD_MODEL_TRAIN_TYPES))

    form.addParam('modelFit', params.EnumParam, default=0,
                  choices=['resnet8', 'resnet16', 'conv31', 'conv63', 'conv127'],
                  condition='modelInitialization== %s' % self.ADD_MODEL_TRAIN_NEW,
                  label='CNN model',
                  help='Model type to fit.\n Your particle must have '
                       'a diameter (longest dimension) after '
                       'downsampling of:\n\n'
                       '<= 70px for resnet8\n'
                       '<= 30px for conv31\n'
                       '<= 62px for conv63\n'
                       '<= 126px for conv127\n')
    form.addParam('prevTopazModel', params.PointerParam,
                  pointerClass='TopazModel',
                  condition='modelInitialization== %s' % self.ADD_MODEL_TRAIN_MODEL, allowsNull=True,
                  label='Select topaz model',
                  help='Select a topaz model to continue from.')

    form.addParam('micsForTraining', params.IntParam,
                  label='Micrographs for training', default=5,
                  help='This number will be divided into training and test data.'
                       'If it is not reached wait')

    form.addSection('Train')
    form.addParam('radius', params.IntParam, default=3,
                  label='Particle radius (px)',
                  allowsPointers=True,
                  help='Pixel radius around particle centers to '
                       'consider.')
    form.addParam('autoenc', params.FloatParam, default=0.,
                  label='Autoencoder',
                  help='Augment the method with autoencoder '
                       'where the weight is on reconstruction error.')
    form.addParam('numEpochs', params.IntParam, default=10,
                  label='Number of epochs',
                  help='Number of training epochs.')
    form.addParam('method', params.EnumParam, default=2,
                  expertLevel=cons.LEVEL_ADVANCED,
                  choices=['PN', 'GE-KL', 'GE-binomial', 'PU'],
                  label='Method',
                  help='Objective function to use for learning the '
                       'region classifier.')
    form.addParam('numPartPerImg', params.IntParam, default=300,
                  allowsPointers=True,
                  expertLevel=cons.LEVEL_ADVANCED,
                  label='Number of particles per image',
                  help='Expected number of particles per micrograph.')
    form.addParam('kfold', params.IntParam, default=5,
                  expertLevel=cons.LEVEL_ADVANCED,
                  label='K-fold',
                  help='Number of subsets to divide the training '
                       'micrographs into. This will determine the '
                       'train/test dataset sizes. E.g. *5* splits the '
                       'picks into five micrograph subsets where one '
                       'will be used as the test dataset; '
                       '20% will be held-out for validation')
    form.addParam('trainExtra', params.StringParam, default='',
                  expertLevel=cons.LEVEL_ADVANCED,
                  label="Advanced options",
                  help="Provide advanced command line options here.")

    form.addParallelSection(threads=1, mpi=1)
    self._definePreprocessParams(form)
    self._defineStreamingParams(form)

    form.getParam('streamingBatchSize').setDefault(32)

  # -------------------------- INSERT steps functions -----------------------
  def _insertAllSteps(self):
    self._defineFileDict()
    ids = [self._insertFunctionStep('convertInputStep',
                                    self.inputCoordinates.getObjId(),
                                    self.scale.get(),
                                    self.kfold.get())]
    if self.doDenoise:
      ids += [self._insertFunctionStep('denoiseStep')]

    ids += [self._insertFunctionStep('preprocessStep')]

    # Training selected
    ids += [self._insertFunctionStep('trainingStep',
                                     self.radius.get(),
                                     self.autoenc.get(),
                                     self.numEpochs.get(),
                                     self.getNNModelFn(),
                                     self.getEnumText('method'),
                                     self.numPartPerImg.get(),
                                     self.trainExtra.get())]

    ids += [self._insertFunctionStep("createOutputStep")]

  def _defineFileDict(self):
    """ Centralize how files are called for iterations and references. """
    trainingFolder = self._getTmpPath("training")
    traindenoiseFolder = os.path.join(trainingFolder, "denoise")
    trainpreFolder = os.path.join(trainingFolder, "preprocess")

    myDict = {
      TRAINING: trainingFolder,
      TRAINING_MIC: os.path.join(trainingFolder, '%(mic)s.mrc'),
      TRAININGDENOISE: traindenoiseFolder,
      TRAININGPREPROCESS: trainpreFolder,
      TRAININGPRE_MIC: os.path.join(trainpreFolder, '%(mic)s.mrc'),
      TRAININGLIST: os.path.join(trainpreFolder, 'image_list_train.txt'),
      TRAININGTEST: os.path.join(trainpreFolder, 'image_list_test.txt'),
      PARTICLES_TRAIN_TXT: os.path.join(trainpreFolder, 'particles_train_test.txt'),
      PARTICLES_TEST_TXT: os.path.join(trainpreFolder, 'particles_test_test.txt'),
      MODEL_FOLDER: self._getExtraPath("model")
    }

    self._updateFilenamesDict(myDict)

  # --------------------------- STEPS functions ------------------------------

  def convertInputStep(self, inputCoordinates, scale, kfold):
    """ Converts a set of coordinates to box files and binaries to mrc
    if needed. It generates 2 folders 1 for the box files and another for
    the mrc files.
    """

    micIds = []
    coordSet = self.inputCoordinates.get()
    setFn = coordSet.getFileName()
    self.debug("Loading input db: %s" % setFn)

    # Load set of coordinates with a user determined number of coordinates for the training step
    enoughMicrographs = False
    while True:
      coordSet = SetOfCoordinates(filename=setFn)
      coordSet._xmippMd = params.String()
      coordSet.loadAllProperties()

      for micAgg in coordSet.aggregate(["MAX"], "_micId", ["_micId"]):
        micIds.append(micAgg["_micId"])
        if len(micIds) == self.micsForTraining.get():
          enoughMicrographs = True
          break
      if enoughMicrographs:
        break
      else:
        if coordSet.isStreamClosed():
          raise Exception("Input coordinates set is closed and there is not enough data to do the training!!.")
        self.info("Not yet there: %s" % len(micIds))
        import time
        time.sleep(10)

    # Create input folder and pre-processed micrographs folder
    micDir = self._getFileName(TRAINING)
    pw.utils.makePath(micDir)
    prepDir = self._getFileName(TRAININGPREPROCESS)
    pw.utils.makePath(prepDir)

    ih = ImageHandler()

    # Get a refreshed set of micrographs
    micsFn = self.inputCoordinates.get().getMicrographs().getFileName()
    # not updating, refresh problem
    coordMics = SetOfMicrographs(filename=micsFn)
    coordMics.loadAllProperties()

    # Create a 0/1 list to mark micrographs for training/testing
    n = len(micIds)
    indexes = np.zeros(n, dtype='int8')
    testSetImages = int((kfold / float(100)) * n)

    # Both the training and the test data set should contain at least one micrograph
    if testSetImages < 1:
      requiredMinimumPercentage = (1 * 100 / n) + 1
      testSetImages = int((requiredMinimumPercentage / float(100)) * n)
    elif testSetImages == n:
      testSetImages = int(0.99 * n)
    indexes[:testSetImages] = 1
    np.random.shuffle(indexes)
    self.info('indexes: %s' % indexes)

    # Write micrographs files
    csvMics = [
      CsvMicrographList(self._getFileName(TRAININGLIST), 'w'),
      CsvMicrographList(self._getFileName(TRAININGTEST), 'w')
    ]

    # Store the micId and indexes in micDict
    micDict = {}
    for i, micId in zip(indexes, micIds):
      mic = coordMics[micId]
      micFn = mic.getFileName()
      baseFn = micId2MicName(micId)
      inputFn = self._getFileName(TRAINING_MIC, **{"mic": baseFn})
      if micFn.endswith('.mrc'):
        pwutils.createAbsLink(os.path.abspath(micFn), inputFn)
      else:
        ih.convert(micFn, inputFn)

      prepMicFn = self._getFileName(TRAININGPRE_MIC, **{"mic": baseFn})

      csvMics[i].addMic(micId, prepMicFn)
      micDict[micId] = i  # store if train or test

    for csv in csvMics:
      csv.close()

    # Write particles files
    csvParts = [
      CsvCoordinateList(self._getFileName(PARTICLES_TRAIN_TXT), 'w'),
      CsvCoordinateList(self._getFileName(PARTICLES_TEST_TXT), 'w')
    ]

    for coord in coordSet.iterItems(orderBy='_micId'):
      micId = coord.getMicId()
      if micId in micDict:
        x = int(round(float(coord.getX()) / scale))
        y = int(round(float(coord.getY()) / scale))
        csvParts[micDict[micId]].addCoord(micId, x, y)

    for csv in csvParts:
      csv.close()

  def denoiseStep(self):
    inputDir = self._getFileName(TRAINING)
    outputDir = self._getFileName(TRAININGDENOISE)
    pwutils.makePath(outputDir)

    args = self.getDenoiseArgs(inputDir, outputDir)
    Plugin.runTopaz(self, 'topaz denoise', args)

  def preprocessStep(self):
    """ Downsamples the micrographs with a factor determined
    by the scale parameter and normalize them with the per-micrograph
    scaled Gaussian mixture model"""

    if self.doDenoise:
      inputDir = self._getFileName(TRAININGDENOISE)
    else:
      inputDir = self._getFileName(TRAINING)
    pwutils.makePath(inputDir)
    outputDir = self._getFileName(TRAININGPREPROCESS)
    pwutils.makePath(outputDir)

    args = self.getPreprocessArgs(inputDir, outputDir)
    Plugin.runTopaz(self, 'topaz preprocess', args)

  def trainingStep(self, radius, enc, numEpochs, modelFit,
                   method, numParts, extra):
    """ Train the model with the provided parameters and the previously
    preprocessed micrograph images and the provided input coordinates.
    """
    outputDir = self._getFileName(MODEL_FOLDER)
    pw.utils.makePath(outputDir)

    args = ' --radius %d' % radius
    args += ' --autoencoder %f' % enc
    args += ' --num-epochs %d' % numEpochs
    args += ' --model %s' % modelFit
    args += ' --method %s' % method
    args += ' --num-particles %d' % numParts
    args += ' --train-images %s' % self._getFileName(TRAININGLIST)
    args += ' --train-targets %s' % self._getFileName(PARTICLES_TRAIN_TXT)
    args += ' --test-images %s' % self._getFileName(TRAININGTEST)
    args += ' --test-targets %s' % self._getFileName(PARTICLES_TEST_TXT)
    args += ' --num-workers %d' % self.numberOfThreads
    args += ' --device %s' % self.gpuList
    args += ' --save-prefix %s/model' % outputDir
    args += ' -o %s/model_training.txt' % outputDir

    if extra != '':
      args += ' ' + extra

    Plugin.runTopaz(self, 'topaz train', args)

    self.MODEL = self.getLastEpochModel(outputDir)

  def createOutputStep(self):
    """ Register the output model. """
    self._defineOutputs(outputModel=TopazModel(self.getOutputModelPath()))

  # --------------------------- UTILS functions --------------------------
  def getPickingFileName(self, micList, key):
    return self._getFileName(key, **{"min": micList[0].strId(),
                                     'max': micList[-1].strId()})

  def getNNModelFn(self):
    '''Returns the model fn (or type) as expected from topaz software'''
    if self.modelInitialization.get() == self.ADD_MODEL_TRAIN_MODEL and self.prevTopazModel.get() != None:
      prevModel = self.prevTopazModel.get()
      return prevModel.getPath()
    else:
      return self.getEnumText('modelFit')

