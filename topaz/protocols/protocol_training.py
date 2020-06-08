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
from pwem.protocols import ProtParticlePickingAuto
from pwem.objects import SetOfMicrographs, SetOfCoordinates
from pwem.emlib.image import ImageHandler

from topaz import convert, Plugin
from topaz.convert import (CsvMicrographList, CsvCoordinateList,
                           readSetOfCoordinates)

TOPAZ_COORDINATES_FILE = 'topaz_coordinates_file'
PICKING_DENOISE_FOLDER = 'picking_denoise_folder'
PICKING_PRE_FOLDER = 'picking_pre_folder'
PICKING_FOLDER = 'picking_folder'
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


class TopazProtTraining(ProtParticlePickingAuto):
    """ Train the Topaz parameters for a picking """
    _label = 'training'

    def __init__(self, **args):
        ProtParticlePickingAuto.__init__(self, **args)
        self.stepsExecutionMode = cons.STEPS_PARALLEL

    # -------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        ProtParticlePickingAuto._defineParams(self, form)

        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      label='Input coordinates', important=True,
                      help='Select the SetOfCoordinates to be used for '
                           'training.')
        form.addParam('boxSize', params.IntParam, default=100,
                      label='Box size (px)', help='Box size in pixels.')
        form.addParam('micsForTraining', params.IntParam,
                      label='Micrographs for training', default=5,
                      help='This number will be divided into training and test data.'
                           'If it is not reached wait')

        form.addSection('Pre-process')
        group = form.addGroup('Denoise')
        group.addParam('doDenoise', params.BooleanParam, default=False,
                       label="Denoise micrographs?")
        group.addParam('modelDenoise', params.EnumParam, default=0,
                       condition='doDenoise',
                       choices=['unet', 'unet-small', 'fcnn', 'affineresnet8'],
                       label='Model',
                       help='Denoising model to use on micrographs.')
        group.addParam('denoiseExtra', params.StringParam, default='',
                       expertLevel=cons.LEVEL_ADVANCED,
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

        form.addSection('Train')
        form.addParam('radius', params.IntParam, default=0,
                      label='Particle radius (px)',
                      help='Pixel radius around particle centers to '
                           'consider.')
        form.addParam('autoenc', params.FloatParam, default=0.,
                      label='Autoencoder',
                      help='Augment the method with autoencoder '
                           'where the weight is on reconstruction error.')
        form.addParam('numEpochs', params.IntParam, default=10,
                      label='Number of epochs',
                      help='Number of training epochs.')
        form.addParam('modelFit', params.EnumParam, default=0,
                      expertLevel=cons.LEVEL_ADVANCED,
                      choices=['resnet8', 'conv31', 'conv63', 'conv127'],
                      label='CNN model',
                      help='Model type to fit.\n Your particle must have '
                           'a diameter (longest dimension) after '
                           'downsampling of:\n\n'
                           '<= 70px for resnet8\n'
                           '<= 30px for conv31\n'
                           '<= 62px for conv63\n'
                           '<= 126px for conv127\n')
        form.addParam('method', params.EnumParam, default=2,
                      expertLevel=cons.LEVEL_ADVANCED,
                      choices=['PN', 'GE-KL', 'GE-binomial', 'PU'],
                      label='Method',
                      help='Objective function to use for learning the '
                           'region classifier.')
        form.addParam('numPartPerImg', params.IntParam, default=300,
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

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                       expertLevel=cons.LEVEL_ADVANCED,
                       label="Choose GPU IDs",
                       help="GPU may have several cores. Set it to zero"
                            " if you do not know what we are talking about."
                            " First core index is 0, second 1 and so on.")

        form.addParallelSection(threads=1, mpi=1)

        self._defineStreamingParams(form)

        form.getParam('streamingBatchSize').setDefault(32)

    # -------------------------- INSERT steps functions -----------------------
    def _insertInitialSteps(self):
        self._defineFileDict()
        ids = [self._insertFunctionStep('convertInputStep',
                                        self.inputCoordinates.getObjId(),
                                        self.scale.get(),
                                        self.kfold.get())]
        if self.doDenoise:
            ids += [self._insertFunctionStep('denoiseStep',
                                             self.getEnumText('modelDenoise'),
                                             self.denoiseExtra.get())]

        ids += [self._insertFunctionStep('preprocessStep',
                                         self.scale.get(),
                                         self.preExtra.get())]

        ids += [self._insertFunctionStep('trainingStep',
                                         self.radius.get(),
                                         self.autoenc.get(),
                                         self.numEpochs.get(),
                                         self.getEnumText('modelFit'),
                                         self.getEnumText('method'),
                                         self.numPartPerImg.get(),
                                         self.trainExtra.get())]
        return ids

    def _defineFileDict(self):
        """ Centralize how files are called for iterations and references. """
        trainingFolder = self._getTmpPath("training")
        traindenoiseFolder = os.path.join(trainingFolder, "denoise")
        trainpreFolder = os.path.join(trainingFolder, "preprocess")

        pickingFolder = self._getTmpPath("micrographs%(min)s-%(max)s")
        pickingDenoiseFolder = os.path.join(pickingFolder, "denoise")
        pickingPreFolder = os.path.join(pickingFolder, "preprocess")
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
            MODEL_FOLDER: os.path.join(trainpreFolder, "model"),
            PICKING_FOLDER: pickingFolder,
            PICKING_DENOISE_FOLDER: pickingDenoiseFolder,
            PICKING_PRE_FOLDER: pickingPreFolder,
            TOPAZ_COORDINATES_FILE: os.path.join(pickingPreFolder,
                                                 "topaz_coordinates%(min)s-%(max)s.txt")
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
                    raise Exception("We have a problem!!")
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
            baseFn = pw.utils.removeBaseExt(micFn)
            inputFn = self._getFileName(TRAINING_MIC, **{"mic": baseFn})
            if micFn.endswith('.mrc'):
                pwutils.createLink(micFn, inputFn)
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

    def denoiseStep(self, modelNoise, extra):
        inputDir = self._getFileName(TRAINING)
        outputDir = self._getFileName(TRAININGDENOISE)
        pwutils.makePath(outputDir)

        args = self.getDenoiseArgs(inputDir, outputDir)
        Plugin.runTopaz(self, 'topaz denoise', args)

    def preprocessStep(self, scale, extra):
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

    def _pickMicrograph(self, micrograph, *args):
        """Picking the given micrograph. """
        self._pickMicrographList([micrograph], *args)

    def _pickMicrographList(self, micList, *args):
        # Link or convert the whole set of micrographs to "batch" folders
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
        boxSize = self.boxSize.get()
        numEpochs = self.numEpochs.get()
        modelDir = self._getFileName(MODEL_FOLDER)

        # Launch process called extract which is rather a prediction
        extractRadius = (boxSize / 2) / self.scale.get()
        args = ' -r %d' % extractRadius
        args += ' -m %s/model_epoch%d.sav' % (modelDir, numEpochs)
        args += ' -o %s' % self.getPickingFileName(micList,
                                                   TOPAZ_COORDINATES_FILE)
        args += ' --num-workers %d' % self.numberOfThreads
        args += ' --device %s' % self.gpuList
        args += ' %s/*.mrc' % preprocessedDir
        Plugin.runTopaz(self, 'topaz extract', args)

    def readCoordsFromMics(self, outputDir, micDoneList, outputCoords):
        """ Read the coordinates from a given list of micrographs """

        outputParticlesFn = self.getPickingFileName(micDoneList,
                                                    TOPAZ_COORDINATES_FILE)

        scale = self.scale.get()
        readSetOfCoordinates(outputParticlesFn, outputCoords.getMicrographs(),
                             outputCoords, scale)

        boxSize = self.boxSize.get()
        outputCoords.setBoxSize(boxSize)

    # --------------------------- UTILS functions --------------------------
    def getPickingFileName(self, micList, key):
        return self._getFileName(key, **{"min": micList[0].strId(),
                                         'max': micList[-1].strId()})

    def getDenoiseArgs(self, inputDir, outDir):
        args = ' %s/*.mrc -o %s/' % (inputDir, outDir)
        args += ' --model %s' % self.getEnumText('modelDenoise')
        args += ' --device %s' % self.gpuList

        if self.denoiseExtra.hasValue():
            args += ' ' + self.denoiseExtra.get()
        else:
            args += ' --normalize'

        return args

    def getPreprocessArgs(self, inputDir, outDir):
        args = " %s/*.mrc -o %s/" % (inputDir, outDir)
        args += " --scale %d " % self.scale.get()
        args += ' --num-workers %d' % self.numberOfThreads
        args += ' --device %s' % self.gpuList

        if self.preExtra.hasValue():
            args += ' ' + self.preExtra.get()

        return args
