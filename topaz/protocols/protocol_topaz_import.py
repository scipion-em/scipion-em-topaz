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
    Imports an existing Topaz training model so that it can be incorporated into a cryo-EM
    processing workflow as a reusable learning resource. The protocol allows previously generated
    Topaz models to become part of the current project environment, making them available for later
    particle-picking or additional model refinement.

    AI Generated:

    Import Training Model (TopazProtImport) — User Manual
        Overview

        The Import Training Model protocol is designed for cryo-EM workflows in which a Topaz model
        has already been trained outside the current project or in a previous processing session.
        Its purpose is to make that trained model available as a formal project object so it can be
        reused in later stages of automated particle analysis.

        In practical biological work, model reuse is highly valuable because training a robust
        particle-picking model often represents a substantial investment of time, annotation effort,
        and empirical optimization. Rather than repeating that effort for every project, users can
        bring previously validated models into a new analysis context.

        Biological Motivation

        Particle-picking models often capture biologically meaningful image characteristics that are
        shared across related datasets. For example, micrographs acquired from the same specimen,
        similar biochemical preparations, or closely related complexes may benefit from the reuse of
        an existing trained model.

        In such cases, importing a prior model can provide a strong starting point for particle
        detection. This often accelerates early workflow stages and reduces the amount of manual
        annotation required to begin productive analysis.

        Continuity Across Projects

        Cryo-EM investigations commonly evolve across multiple processing sessions, projects, or
        collaborative environments. A training model developed during one stage of a study may
        remain valuable long after the original project has finished.

        This protocol supports that continuity by allowing trained Topaz models to persist as
        transferable analytical assets. From a practical standpoint, this is especially useful in
        facility environments, long-term structural biology programs, or collaborative studies where
        standardized particle-picking behavior is desirable.

        Reuse for Further Training

        Imported models are not limited to direct particle-picking applications. They may also serve
        as starting points for additional training when new micrographs become available or when a
        dataset evolves.

        Biologically, this is important because datasets often expand gradually. New imaging
        conditions, new sample preparations, or related conformational states may require model
        adaptation rather than full retraining. Beginning from a previously learned representation
        often improves efficiency and can help preserve useful prior knowledge.

        Practical Considerations

        The imported model should ideally be relevant to the target biological system and image
        characteristics of the current project. Models trained on unrelated particle types, very
        different acquisition conditions, or strongly different contrast regimes may provide limited
        benefit.

        In practice, the greatest advantage is usually obtained when the imported model originates
        from data that are biologically or experimentally similar to the current dataset.

        Portability and Project Management

        Because the protocol incorporates an external model into the project structure, it provides
        a convenient organizational layer for model management. This makes the imported model easier
        to track, reference, and reuse across later processing stages.

        Users should nevertheless keep in mind that external model resources remain part of a larger
        computational environment. Good project organization and careful data management remain
        important for long-term reproducibility.

        Final Perspective

        For cryo-EM users, importing a trained model is not merely a file-management task but a way
        of preserving learned information from previous analyses. A well-performing Topaz model
        represents accumulated knowledge about particle appearance, contrast behavior, and dataset
        characteristics.

        By making prior training outcomes immediately available for reuse, this protocol helps
        connect independent projects into a more efficient and biologically consistent particle
        analysis workflow.
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

