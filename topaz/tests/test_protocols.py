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



import pyworkflow as pw
import pyworkflow.em
from topaz.protocols import TopazProtTraining


XmippProtPreprocessMicrographs = pw.utils.importFromPlugin(
    'xmipp3.protocols', 'XmippProtPreprocessMicrographs', doRaise=True)


class TestTopaz(pw.tests.BaseTest):
    """ Test cryolo protocol"""

    @classmethod
    def setData(cls):
        cls.ds = pw.tests.DataSet.getDataSet('relion_tutorial')

    @classmethod
    def setUpClass(cls):
        pw.tests.setupTestProject(cls)
        cls.setData()

    def runImportMicrograph(self):

        """ Run an Import micrograph protocol. """
        protImport = self.newProtocol(pw.em.ProtImportMicrographs,
                                         samplingRateMode=0,
                                         filesPath=self.ds.getFile('micrographs/*.mrc'),
                                         samplingRate=3.54,
                                         magnification=59000,
                                         voltage=300,
                                         sphericalAberration=2)

        self.launchProtocol(protImport)
        self.assertSetSize(protImport.outputMicrographs, 20,
                           "There was a problem with the import")

        return protImport

    def runMicPreprocessing(self, inputMics):

        print "Preprocessing the micrographs..."
        protPreprocess = self.newProtocol(XmippProtPreprocessMicrographs,
                                          doCrop=True, cropPixels=25)
        self.protPreprocess = protPreprocess
        protPreprocess.inputMicrographs.set(inputMics)
        protPreprocess.setObjLabel('crop 50px')
        self.launchProtocol(protPreprocess)
        self.assertSetSize(protPreprocess.outputMicrographs, 20,
                           "There was a problem with the preprocessing")

        return protPreprocess

    def runImportCoords(self, inputMics):
        """ Run an Import coords protocol. """
        protImportCoords = self.newProtocol(pw.em.ProtImportCoordinates,
                                            importFrom=pw.em.ProtImportCoordinates.IMPORT_FROM_EMAN,
                                            objLabel='import EMAN coordinates',
                                            filesPath=self.ds.getFile('pickingEman/info/'),
                                            inputMicrographs=inputMics,
                                            filesPattern='*.json',
                                            boxSize=65)
        self.launchProtocol(protImportCoords)
        self.assertSetSize(protImportCoords.outputCoordinates,
                           msg="There was a problem importing eman coordinates")
        return protImportCoords

    def test_training(self):
        # Run needed protocols
        protImportMics = self.runImportMicrograph()
        protPrep = self.runMicPreprocessing(protImportMics.outputMicrographs)
        protImportCoords = self.runImportCoords(protPrep.outputMicrographs)
        inputCoords = protImportCoords.outputCoordinates
        protTopazTrain = self.newProtocol(TopazProtTraining,
                                          objLabel='topaz - training',
                                          inputMicrographs=self.protPreprocess.outputMicrographs,
                                          micsForTraining=10,
                                          inputCoordinates=inputCoords,
                                          splitData=50,
                                          boxSize=65,
                                          scale=4,
                                          radius=0,
                                          pi=0.035,
                                          model=1,  # conv31
                                          numEpochs=2,
                                          streamingBatchSize=4)

        self.launchProtocol(protTopazTrain)

