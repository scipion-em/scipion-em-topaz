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

        group = form.addGroup('Training')

        group.addParam('radius', params.IntParam, default=0,
                       label='Radius (px)',
                       help='Radius (in pixels) around particle centers to '
                            'consider positive. ')

        # FIXME: Check the default of the following parameter
        group.addParam('pi', params.FloatParam, default=0.035,
                       label='Pi',
                       help='Parameter specifying fraction of data that is '
                            'expected to be positive')

        # FIXME: Check all possible models
        group.addParam('model', params.EnumParam, default=0,
                       choices=['resnet8', 'conv31'],
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
                                 self.scale.get())
        self._insertFunctionStep('trainingStep',
                                 self.radius.get(),
                                 self.pi.get(),
                                 self.getEnumText('model'))
        self._insertFunctionStep('createOutputStep',
                                 self.boxSize.get())
        self._insertFunctionStep('createOutputStep2',
                                 np.random.rand())

    # --------------------------- STEPS functions ------------------------------
    def _getInputPath(self, *paths):
        return self._getExtraPath(*paths)

    def convertInputStep(self, coordsId, scale):
        """ Converts a set of coordinates to box files and binaries to mrc
        if needed. It generates 2 folders 1 for the box files and another for
        the mrc files. To be passed (the folders as params for cryolo
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
        testN = int(n/10)  # fixme: use percent input var
        indexes[:testN] = 1
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

            prepMicFn = self._getInputPath('Preprocessed', '%s.tiff' % baseFn)
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
            x = coord.getX() / scale
            y = coord.getY() / scale
            csvParts[micDict[micId]].addCoord(micId, x, y)

        for csv in csvParts:
            csv.close()

        self.runTopaz('preprocess -s%d %s/*.mrc -o %s/'
                      % (scale, micDir, prepDir))

    def trainingStep(self, radius, pi, model):
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
        args += ' --save-prefix=%s/model' % outputDir
        args += ' -o %s/model_training.txt' % outputDir

        self.runTopaz('train %s' % args)

    def createOutputStep(self, boxSize):
        """ Run the topaz extract command and convert the resulting coordinates,
        taking into account the scaling of the original input micrographs.

        topaz extract -r7 -m saved_models/EMPIAR-10025/model_epoch10.sav \
              -o saved_models/EMPIAR-10025/predicted_particles_all.txt \
              data/EMPIAR-10025/processed/micrographs/*.tiff
        """
        prepDir = self._getInputPath('Preprocessed')
        outputDir = self._getExtraPath('saved_models')

        # TODO: Maybe allow as parameter? It should be clear for the user if relative
        # to input micrographs or to the downscaled ones
        extractRadius = (boxSize / 4) / self.scale.get()
        args = '-r%d' % extractRadius
        args += ' -m %s/model_epoch10.sav' % outputDir  #TODO: Check this if more than 10 epochs
        args += ' -o %s/predicted_particles_all.txt' % outputDir
        args += ' %s/*.tiff' % prepDir

        self.runTopaz('extract %s' % args)

    def createOutputStep2(self, randAlways):
        inputMics = self.inputCoordinates.get().getMicrographs()
        outCoordSet = self._createSetOfCoordinates(inputMics)
        outputParticlesFn = self._getExtraPath('saved_models',
                                               'predicted_particles_all.txt')
        readSetOfCoordinates(outputParticlesFn, inputMics, outCoordSet)
        self._defineOutputs(outputCoordinates=outCoordSet)


    # --------------------------- INFO functions ------------------------------
    def _validate(self):
        errors = []
        return errors
