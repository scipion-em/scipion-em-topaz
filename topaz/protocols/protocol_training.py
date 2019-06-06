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

import numpy as np
from itertools import izip

import pyworkflow as pw
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons

from .protocol_base import TopazProtocol

from topaz.convert import (CsvMicrographList, CsvCoordinateList,
                           readSetOfCoordinates)


class TopazProtTraining(pw.em.ProtParticlePicking, TopazProtocol):
    """ Train the Topaz parameters for a picking
    """
    _label = 'training'

    # -------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputCoordinates', params.PointerParam,
                      pointerClass='SetOfCoordinates',
                      label='Input coordinates', important=True,
                      help='Select the SetOfCoordinates to be used for '
                           'training.')

        form.addParam('boxSize', params.IntParam, default=100,
                      label='Box Size', help='Box size in pixels')

        group = form.addGroup('Pre-processing')

        group.addParam('scale', params.IntParam, default=4,
                       label='Scale',
                       help='Factor to down-scale the micrographs for '
                            'pre-processing')

        group.addParam('splitData', params.IntParam, default=10,
                      label='Split data',
                      help='Data is split into train and test sets. '
                           'E.g.:10 means 10% of the input micrographs and associated labeled '
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

        form.addParallelSection(threads=4, mpi=0)

    # -------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('convertInputStep',
                                 self.inputCoordinates.getObjId(),
                                 self.scale.get(),
                                 self.splitData.get())
        self._insertFunctionStep('trainingStep',
                                 self.radius.get(),
                                 self.numEpochs.get(),
                                 self.pi.get(),
                                 self.getEnumText('model'))
        self._insertFunctionStep('predictStep',
                                 self.boxSize.get(),
                                 self.numEpochs.get())
        self._insertFunctionStep('createOutputStep')

    # --------------------------- STEPS functions ------------------------------
    def _getInputPath(self, *paths):
        return self._getExtraPath(*paths)

    def convertInputStep(self, inputCoordinates, scale, splitData):
        """ Converts a set of coordinates to box files and binaries to mrc
        if needed. It generates 2 folders 1 for the box files and another for
        the mrc files. To be passed (the folders as params for cryolo)
        """

        coordSet = self.inputCoordinates.get()
        coordMics = coordSet.getMicrographs()

        # Create input folder for pre-processed micrographs
        micDir = self._getInputPath('Micrographs')
        pw.utils.makePath(micDir)
        prepDir = self._getInputPath('Preprocessed')
        pw.utils.makePath(prepDir)

        ih = pw.em.ImageHandler()

        # Create a 0/1 list to mark micrographs for training/testing
        n = coordMics.getSize()
        indexes = np.zeros(n, dtype='int8')
        testSetImages = int((float(splitData) / 100) * n)
        indexes[:testSetImages] = 1
        np.random.shuffle(indexes)
        self.info('indexes: %s' % indexes)

        # Write micrographs files
        csvMics = [
            CsvMicrographList(self._getInputPath('image_list_train.txt'), 'w'),
            CsvMicrographList(self._getInputPath('image_list_test.txt'), 'w')
        ]

        micDict = {}

        for i, mic in izip(indexes, coordMics):
            fn = mic.getFileName()
            baseFn = pw.utils.removeBaseExt(fn)
            inputFn = self._getInputPath('Micrographs', '%s.mrc' % baseFn)
            if fn.endswith('.mrc'):
                pw.utils.createLink(fn, inputFn)
            else:
                ih.convert(mic, inputFn)

            prepMicFn = self._getInputPath('Preprocessed', '%s.mrc' % baseFn)
            csvMics[i].addMic(mic.getObjId(), prepMicFn)
            micDict[mic.getObjId()] = i  # store if train or test

        for csv in csvMics:
            csv.close()

        # Write particles files
        csvParts = [
            CsvCoordinateList(self._getInputPath('particles_train.txt'), 'w'),
            CsvCoordinateList(self._getInputPath('particles_test.txt'), 'w')
        ]

        for coord in coordSet:
            micId = coord.getMicId()
            x = int(round(float(coord.getX()) / scale))
            y = int(round(float(coord.getY()) / scale))
            csvParts[micDict[micId]].addCoord(micId, x, y)

        for csv in csvParts:
            csv.close()

        self.runTopaz('preprocess -s%d %s/*.mrc -o %s/'
                      % (scale, micDir, prepDir))

    def trainingStep(self, radius, numEpochs, pi, model):
        """ Train the model with the provided parameters and the previously
        preprocessed micrograph images and the provided input coordinates.
        """
        outputDir = self._getExtraPath('saved_models')
        pw.utils.makePath(outputDir)

        args = ' --radius %d' % radius
        args += ' --pi %f' % pi
        args += ' --model %s' % model
        args += ' --train-images %s' % self._getInputPath('image_list_train.txt')
        args += ' --train-targets %s' % self._getInputPath('particles_train.txt')
        args += ' --test-images %s' % self._getInputPath('image_list_test.txt')
        args += ' --test-targets %s' % self._getInputPath('particles_test.txt')
        args += ' --num-workers=%d' % self.numberOfThreads
        args += ' --device %s' % self.gpuList
        args += ' --num-epochs %d' % numEpochs
        args += ' --save-prefix=%s/model' % outputDir
        args += ' -o %s/model_training.txt' % outputDir

        self.runTopaz('train %s' % args)

    def predictStep(self, boxSize, numEpochs):
        """ Run the topaz extract command and convert the resulting coordinates,
        taking into account the scaling of the original input micrographs.
        """
        prepDir = self._getInputPath('Preprocessed')
        outputDir = self._getExtraPath('saved_models')

        # TODO: Maybe allow as parameter? It should be clear for the user if relative
        # to input micrographs or to the downscaled ones
        extractRadius = (boxSize / 2) / self.scale.get()
        args = '-r%d' % extractRadius
        args += ' -m %s/model_epoch%d.sav' % (outputDir, numEpochs)
        args += ' -o %s/predicted_particles_all.txt' % outputDir
        args += ' %s/*.mrc' % prepDir

        self.runTopaz('extract %s' % args)

    def createOutputStep(self):
        inputMics = self.inputCoordinates.get().getMicrographs()
        outCoordSet = self._createSetOfCoordinates(inputMics)
        outCoordSet.copyInfo(self.inputCoordinates.get())
        outputParticlesFn = self._getExtraPath('saved_models',
                                               'predicted_particles_all.txt')
        readSetOfCoordinates(outputParticlesFn, inputMics, outCoordSet, self.scale.get())
        self._defineOutputs(outputCoordinates=outCoordSet)


    # --------------------------- INFO functions -------------------------------
    def _validate(self):
        validateMsgs = []

        coordSet = self.inputCoordinates.get()
        coordMics = coordSet.getMicrographs()
        n = coordMics.getSize()

        testSetImages = int((float(self.splitData.get()) / 100) * n)
        if testSetImages == 0:
            requiredMinPercentage = ((1 * 100) / n) + 1
            validateMsgs.append("Please set splitData to minimum: %d" % requiredMinPercentage)
        elif testSetImages == n:
            validateMsgs.append("Please set splitData to maximum: 99")

        return validateMsgs


