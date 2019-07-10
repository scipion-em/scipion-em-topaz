# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
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
from itertools import izip

import pyworkflow as pw
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons
from pyworkflow.em import ProtParticlePickingAuto, pwutils, SetOfMicrographs, SetOfCoordinates

from topaz import convert
from .protocol_base import TopazProtocol

from topaz.convert import (CsvMicrographList, CsvCoordinateList,
                           readSetOfCoordinates)


TOPAZ_COORDINATES_FILE = 'topaz_coordinates_file'

PICKING_PRE_FOLDER = 'picking_pre_folder'

PICKING_FOLDER = 'picking_folder'

MODEL_FOLDER = 'model_folder'
TRAINING = 'training'
TRAINING_MIC = 'trainingMic'
TRAININGPREPROCESS = 'trainingpreprocess'
TRAININGPRE_MIC = 'trainingpreMic'
TRAININGLIST = 'traininglist'
TRAININGTEST = 'trainingtest'
PARTICLES_TEST_TXT = 'particles_test.txt'
PARTICLES_TRAIN_TXT = 'particles_train.txt'


class TopazProtTraining(pw.em.ProtParticlePickingAuto, TopazProtocol):
    """ Train the Topaz parameters for a picking
    """
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
                      label='Box Size', help='Box size in pixels')

        form.addParam('micsForTraining', params.IntParam,
                      label='Micrographs for training', default=5,
                      help='This number will be divided into training and test data.'
                           'If it is not reached wait')


        group = form.addGroup('Pre-processing')

        group.addParam('scale', params.IntParam, default=4,
                       label='Scale',
                       help='Factor to down-scale the micrographs for '
                            'pre-processing')

        group.addParam('splitData', params.IntParam, default=10,
                      label='Split data',
                      help='Data is split into train and test sets. '
                           'E.g.:10 means 10% of the number of micrographs selected for training and associated labeled '
                           'particles will go into the test set, 90% will go into the train set.')


        group = form.addGroup('Training')

        group.addParam('radius', params.IntParam, default=0,
                       label='Radius (px)',
                       help='Radius (in pixels) around particle centers to '
                            'consider positive. ')

        group.addParam('numEpochs', params.IntParam, default=10,
                       label='Number of epochs',
                       help='Maximum number of training epochs')

        group.addParam('pi', params.FloatParam, default=0.035,
                       label='Pi',
                       help='Parameter specifying fraction of data that is '
                            'expected to be positive. Pi<0.05')

        group.addParam('model', params.EnumParam, default=0,
                       choices=['resnet8', 'conv31', 'conv63', 'conv127'],
                       label='Model',
                       help='Model type to fit.')

        form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                         expertLevel=cons.LEVEL_ADVANCED,
                         label="Choose GPU IDs",
                         help="GPU may have several cores. Set it to zero"
                              " if you do not know what we are talking about."
                              " First core index is 0, second 1 and so on."
                              " Motioncor2 can use multiple GPUs - in that case"
                              " set to i.e. *0 1 2*.")

        form.addParallelSection(threads=1, mpi=1)

        self._defineStreamingParams(form)

    # -------------------------- INSERT steps functions -----------------------
    def _insertInitialSteps(self):

        self._defineFileDict()

        id = [self._insertFunctionStep('convertInputStep',
                                     self.inputCoordinates.getObjId(),
                                     self.splitData.get(),
                                     self.scale.get()),
              self._insertFunctionStep('preprocessStep',
                                     self.scale.get()),
            self._insertFunctionStep('trainingStep',
                                     self.radius.get(),
                                     self.numEpochs.get(),
                                     self.pi.get(), self.getEnumText('model'))]
        return id



    def _defineFileDict(self):
        """ Centralize how files are called for iterations and references. """
        trainingFolder = self._getTmpPath("training")
        trainpreFolder = os.path.join(trainingFolder, "preprocess")


        pickingFolder = self._getTmpPath("micrographs%(min)s-%(max)s")
        pickingPreFolder = os.path.join(pickingFolder, "preprocess")
        myDict = {
            TRAINING: trainingFolder,
            TRAINING_MIC: os.path.join(trainingFolder, '%(mic)s.mrc'),
            TRAININGPREPROCESS: trainpreFolder,
            TRAININGPRE_MIC: os.path.join(trainpreFolder, '%(mic)s.mrc'),
            TRAININGLIST: os.path.join(trainpreFolder, 'image_list_train.txt'),
            TRAININGTEST: os.path.join(trainpreFolder, 'image_list_test.txt'),
            PARTICLES_TRAIN_TXT: os.path.join(trainpreFolder, 'particles_train_test.txt'),
            PARTICLES_TEST_TXT: os.path.join(trainpreFolder, 'particles_test_test.txt'),
            MODEL_FOLDER: os.path.join(trainpreFolder, "model"),
            PICKING_FOLDER: pickingFolder,
            PICKING_PRE_FOLDER: pickingPreFolder,
            TOPAZ_COORDINATES_FILE: os.path.join(pickingPreFolder, "topaz_coordinates%(min)s-%(max)s.txt")
        }

        self._updateFilenamesDict(myDict)

    # --------------------------- STEPS functions ------------------------------

    def convertInputStep(self, inputCoordinates, splitData, scale):
        """ Converts a set of coordinates to box files and binaries to mrc
        if needed. It generates 2 folders 1 for the box files and another for
        the mrc files. To be passed (the folders as params for cryolo)
        """

        micIds = []
        coordSet = self.inputCoordinates.get()
        setFn = coordSet.getFileName()
        self.debug("Loading input db: %s" % setFn)

        # Load set of coordinates with a user determined number of coordinates for the training step
        while True:
            coordSet = SetOfCoordinates(filename=setFn)
            coordSet._xmippMd = params.String()
            coordSet.loadAllProperties()

            for micAgg in coordSet.aggregate(["MAX"], "_micId", ["_micId"]):
                micIds.append(micAgg["_micId"])
                if len(micIds) == self.micsForTraining.get():
                    break
            if micAgg["_micId"] == self.micsForTraining.get():
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

        ih = pw.em.ImageHandler()

        # Get a refreshed set of micrographs
        micsFn = self.inputCoordinates.get().getMicrographs().getFileName()   # not updating, refresh problem
        coordMics = SetOfMicrographs(filename=micsFn)
        coordMics.loadAllProperties()

        # Create a 0/1 list to mark micrographs for training/testing
        n = len(micIds)
        indexes = np.zeros(n, dtype='int8')
        testSetImages = int((splitData / float(100)) * n)

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
        for i, micId in izip(indexes, micIds):
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

        for micId in micIds:
            # Loop through the subset of coordinates that was picked by the previous step
            for coord in coordSet.iterItems(orderBy='_micId'):
                x = int(round(float(coord.getX()) / scale))
                y = int(round(float(coord.getY()) / scale))
                csvParts[micDict[micId]].addCoord(micId, x, y)

        for csv in csvParts:
            csv.close()

    def preprocessStep(self, scale):
        """ Downsamples the micrographs with a factor determined
        by the scale parameter and normalize them with the per-micrograph
        scaled Gaussian mixture model"""

        inputDir = self._getFileName(TRAINING)
        pwutils.makePath(inputDir)
        outputDir = self._getFileName(TRAININGPREPROCESS)
        pwutils.makePath(outputDir)
        self.runTopaz('preprocess -s%d %s/*.mrc -o %s/' % (scale, inputDir, outputDir))


    def trainingStep(self, radius, numEpochs, pi, model):
        """ Train the model with the provided parameters and the previously
        preprocessed micrograph images and the provided input coordinates.
        """
        outputDir = self._getFileName(MODEL_FOLDER)
        pw.utils.makePath(outputDir)

        args = ' --radius %d' % radius
        args += ' --pi %f' % pi
        args += ' --model %s' % model
        args += ' --train-images %s' % self._getFileName(TRAININGLIST)
        args += ' --train-targets %s' % self._getFileName(PARTICLES_TRAIN_TXT)
        args += ' --test-images %s' % self._getFileName(TRAININGTEST)
        args += ' --test-targets %s' % self._getFileName(PARTICLES_TEST_TXT)
        args += ' --num-workers=%d' % self.numberOfThreads
        args += ' --device %s' % self.gpuList
        args += ' --num-epochs %d' % numEpochs
        args += ' --save-prefix=%s/model' % outputDir
        args += ' -o %s/model_training.txt' % outputDir

        self.runTopaz('train %s' % args)

    def _pickMicrograph(self, micrograph, *args):
        """Picking the given micrograph. """
        self._pickMicrographList([micrograph], *args)

    def _pickMicrographList(self, micList, *args):
        # Link or convert the whole set of micrographs to "batch" folders
        workingDir = self.getPickingFileName(micList, PICKING_FOLDER)
        pwutils.makePath(workingDir)

        convert.convertMicrographs(micList, workingDir)

        # create preprocessed folder under the workingDir. Now in the extra folder should be replaced in tmp folder
        preprocessedDir = self.getPickingFileName(micList, PICKING_PRE_FOLDER)
        pwutils.makePath(preprocessedDir)

        # preprocess the micrographs in the batch folder, output in preprocessedDir
        self.runTopaz('preprocess -s%d %s/*.mrc -o %s/' % (self.scale.get(), workingDir, preprocessedDir))

        # perform prediction on the preprocessed micrographs
        boxSize = self.boxSize.get()
        numEpochs = self.numEpochs.get()
        modelDir = self._getFileName(MODEL_FOLDER)

        # Launch process called extract which is rather a prediction
        extractRadius = (boxSize / 2) / self.scale.get()
        args = '-r%d' % extractRadius
        args += ' -m %s/model_epoch%d.sav' % (modelDir, numEpochs)
        args += ' -o %s' % self.getPickingFileName(micList, TOPAZ_COORDINATES_FILE)
        args += ' %s/*.mrc' % preprocessedDir

        self.runTopaz('extract %s' % args)


    def readCoordsFromMics(self, outputDir, micDoneList, outputCoords):
        """ Read the coordinates from a given list of micrographs """

        outputParticlesFn = self.getPickingFileName(micDoneList, TOPAZ_COORDINATES_FILE)

        scale = self.scale.get()
        readSetOfCoordinates(outputParticlesFn, outputCoords.getMicrographs(), outputCoords, scale)

        boxSize = self.boxSize.get()
        outputCoords.setBoxSize(boxSize)

    # --------------------------- UTILS functions --------------------------
    def getPickingFileName(self, micList, key):

        return self._getFileName(key, **{"min": micList[0].strId(), 'max': micList[-1].strId()})


