# **************************************************************************
# *
# * Authors:     Daniel Del Hoyo (daniel.delhoyo.gomez@alumnos.upm.es)
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


from pyworkflow.tests import BaseTest, setupTestProject, DataSet
from pyworkflow.plugin import Domain

from pwem.protocols.protocol_import import ProtImportMicrographs, ProtImportCoordinates

import topaz.protocols as protocols

XmippProtPreprocessMicrographs = Domain.importFromPlugin(
    'xmipp3.protocols', 'XmippProtPreprocessMicrographs', doRaise=True)

class TestTopaz(BaseTest):
    """ Test Topaz protocol"""
    @classmethod
    def setData(cls):
        cls.ds = DataSet.getDataSet('relion_tutorial')

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.setData()
        # Run needed protocols
        cls.runImportMicrograph()
        cls.runMicPreprocessing()

    @classmethod
    def runImportMicrograph(cls):

        """ Run an Import micrograph protocol. """
        protImport = cls.newProtocol(
            ProtImportMicrographs,
            samplingRateMode=0,
            filesPath=TestTopaz.ds.getFile('micrographs/*.mrc'),
            samplingRate=3.54,
            magnification=59000,
            voltage=300,
            sphericalAberration=2)

        cls.launchProtocol(protImport)
        cls.protImport = protImport

    @classmethod
    def runMicPreprocessing(cls):

        print("Preprocessing the micrographs...")
        protPreprocess = cls.newProtocol(XmippProtPreprocessMicrographs,
                                         doCrop=True, cropPixels=25)
        protPreprocess.inputMicrographs.set(cls.protImport.outputMicrographs)
        protPreprocess.setObjLabel('crop 50px')
        cls.launchProtocol(protPreprocess)
        cls.protPreprocess = protPreprocess

    @classmethod
    def runImportCoords(cls):
        """ Run an Import coords protocol. """
        protImportCoords = cls.newProtocol(
            ProtImportCoordinates,
            importFrom=ProtImportCoordinates.IMPORT_FROM_EMAN,
            objLabel='import EMAN coordinates',
            filesPath=TestTopaz.ds.getFile('pickingEman/info/'),
            inputMicrographs=cls.protPreprocess.outputMicrographs,
            filesPattern='*.json',
            boxSize=65)
        cls.launchProtocol(protImportCoords)
        cls.protImportCoords = protImportCoords

    def _runTraining(self, modelInit=0, prevModel=None, denoise=False):
        self.runImportCoords()
        # Topaz training
        protTraining = self.newProtocol(
            protocols.TopazProtTraining,
            label='Training 1',
            inputMicrographs=self.protPreprocess.outputMicrographs,
            inputCoordinates=self.protImportCoords.outputCoordinates,
            modelInitialization=modelInit,
            prevTopazModel=prevModel,
            radius=3, scale=4, doDenoise=denoise,
            numEpochs=1)
        self.launchProtocol(protTraining)

        outputModel = getattr(protTraining, 'outputModel', None)
        self.assertTrue(outputModel is not None)

        # Training mode picking
        protPicking = self.newProtocol(
            protocols.TopazProtPicking,
            label="Picking after Training 1",
            inputMicrographs=self.protPreprocess.outputMicrographs,
            prevTopazModel=protTraining.outputModel,
            boxSize=50, doDenoise=denoise,
            streamingBatchSize=10)

        self.launchProtocol(protPicking)
        return protTraining, protPicking

    def _runImportModel(self, prot):
        protImport = self.newProtocol(
            protocols.TopazProtImport,
            label='Importing 1',
            modelPath=prot._getExtraPath('model/model_epoch1.sav'))
        self.launchProtocol(protImport)
        return protImport

    def testPickingNoTraining(self):
        # No training mode picking
        protTopaz = self.newProtocol(
            protocols.TopazProtPicking,
            inputMicrographs=self.protPreprocess.outputMicrographs,
            modelInitialization=1,
            radius=10,
            scale=4,
            boxSize=100,
            streamingBatchSize=10)
        self.launchProtocol(protTopaz)

    def testTraining(self):
        #Training a new model and picking
        protTrained, protPicked = self._runTraining(denoise=True)
        #Importing a model from path
        protImported = self._runImportModel(protTrained)
        #Training an imported model and picking
        self._runTraining(modelInit=1, prevModel=protImported.outputModel)

