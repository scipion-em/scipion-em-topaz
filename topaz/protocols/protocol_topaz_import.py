# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
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

import pyworkflow.utils as pwutils
import pyworkflow.protocol.params as params
from pwem.protocols import ProtImport

from topaz.objects import TopazModel


class TopazProtImport(ProtImport):
    """
    Imports an existing Topaz training model into the project.

    AI Generated:

    Import Training Model (TopazProtImport) — User Manual
        Overview

        The Import Training Model protocol allows an already trained
        Topaz model to be incorporated into the current Scipion project.
        Instead of training a new model from scratch, this protocol
        registers a previously generated Topaz model so it can be reused
        directly for particle picking or as a starting point for further
        training.

        In practical cryo-EM workflows, this is especially useful when
        a model has already been optimized on similar datasets, similar
        particle types, or previous experiments. Reusing trained models
        can save substantial computational time and often improves
        consistency across projects.

        Input Parameter

        The protocol requires a single input:

        Training model path:
            The user provides the filesystem path to an existing Topaz
            training model.

        This file is expected to correspond to a valid Topaz model
        generated previously by a Topaz training protocol or by an
        external workflow.

        Internal Workflow

        Once executed, the protocol performs a simple but important task.

        First, it converts the provided path into an absolute path in
        order to avoid ambiguities related to the current working
        directory.

        Then, it creates a symbolic link inside the protocol working
        directory. The imported model itself is not duplicated, which
        keeps storage usage low and avoids unnecessary copies of
        potentially large files.

        Finally, the linked model is registered as an output object
        inside Scipion, making it available for downstream protocols.

        Output

        After execution, the protocol generates one output:

        outputModel:
            A TopazModel object pointing to the imported training model.

        This output can be used directly by other Topaz-related
        protocols, particularly particle picking or continued training.

        Practical Considerations

        Since the protocol creates a symbolic link rather than copying
        the original file, the imported model remains dependent on the
        original file location.

        If the project is moved to another computer, another filesystem,
        or the original model is deleted or relocated, the symbolic link
        may become invalid.

        In such cases, the model is not lost, but the link must be
        updated manually so the project can access the original file
        again.

        Recommended Usage

        This protocol is especially useful in the following situations:

        - Reusing a previously trained Topaz model for particle picking.
        - Sharing a validated model across multiple projects.
        - Continuing training from an existing checkpoint.
        - Standardizing particle picking across related datasets.

        Final Perspective

        The Import Training Model protocol is intentionally simple, but
        it plays an important role in efficient cryo-EM processing.
        Rather than retraining models repeatedly, users can reuse
        existing Topaz models and integrate them directly into new
        workflows with minimal overhead.
    """
    _label = 'import training model'

    # -------------------------- DEFINE param functions -----------------------
    def _defineParams(self, form):
        form.addSection(label='Import')
        form.addParam('modelPath', params.PathParam,
                      label="Training model path",
                      help="Provide the path of a previous topaz training "
                           "model. ")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep("importModelStep")

    # --------------------------- STEPS functions -----------------------------
    def importModelStep(self):
        """ Create a link to the provided input model path
        and register the output to be used later for further training
        or picking.
        """
        absPath = os.path.abspath(self.modelPath.get())
        outputPath = self._getExtraPath(os.path.basename(absPath))
        self.info("Creating link:\n"
                  "%s -> %s" % (outputPath, absPath))
        self.info("NOTE: If you move this project to another computer, the symbolic"
                  "link to the model will be broken, but you can update the link "
                  "and get it working again. ")

        pwutils.createAbsLink(absPath, outputPath)

        self._defineOutputs(outputModel=TopazModel(outputPath))

