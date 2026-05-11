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
    """
    Performs automated particle picking on cryo-EM micrographs using neural
    network models from the Topaz framework. The protocol is intended to
    identify candidate particle locations directly from micrograph data,
    producing coordinate sets suitable for downstream extraction,
    classification, and reconstruction.

    AI Generated:

    Topaz Particle Picking (TopazProtPicking) — User Manual
        Overview

        The Topaz particle picking protocol provides automated particle
        detection for cryo-electron microscopy micrographs using deep
        learning–based models. Its main objective is to transform raw
        micrograph images into biologically meaningful particle coordinate
        sets, enabling efficient downstream single-particle workflows.
        Instead of relying on manually tuned templates or user-defined
        heuristics, the protocol leverages learned image features to detect
        particles even in difficult datasets with low contrast, uneven ice,
        contamination, or heterogeneous particle appearance.

        For biological users, this protocol is particularly valuable when
        processing large collections of micrographs where manual picking
        would be impractical. It is commonly used both in exploratory
        projects, where rapid feedback is important, and in production
        workflows where reproducibility and throughput are essential.

        Model Selection and Biological Context

        The protocol can operate using either a previously trained Topaz
        model or one of the general pretrained models distributed with the
        Topaz ecosystem. This distinction is biologically important.

        A previously trained model is typically preferable when it was
        generated from particles closely related to the current dataset.
        In such cases, the learned representation often captures specimen-
        specific characteristics such as preferred orientations, particle
        size, ice thickness behavior, or imaging conditions. This usually
        leads to higher precision and more biologically reliable particle
        localization.

        General pretrained models provide a convenient starting point when
        no project-specific training is available. They are especially
        useful during early exploratory work, feasibility assessments, or
        rapid screening of newly acquired datasets. However, because these
        models were trained on broader data, they may not optimally capture
        the morphology of highly unusual particles, elongated assemblies,
        membrane proteins, or strongly flexible complexes.

        Input Micrographs and Preprocessing

        The protocol operates directly on input micrographs. In practical
        cryo-EM work, the quality of these micrographs strongly influences
        the final picking results. Clean micrographs with well-estimated
        CTF parameters and limited contamination generally produce the most
        reliable coordinates.

        Before particle prediction, the protocol may apply denoising and
        preprocessing transformations. These operations are not merely
        computational conveniences—they can substantially affect biological
        interpretability.

        Denoising can improve particle detectability in especially noisy
        datasets. This is often useful for small particles, weakly
        scattering complexes, or micrographs collected at low dose.
        Nevertheless, excessive denoising may alter subtle structural
        features, so users should interpret improvements with caution.

        Preprocessing also performs downsampling. In most biological
        applications this improves robustness by emphasizing larger
        particle-scale features over high-frequency noise. A practical
        consequence is that picking becomes more stable and faster, but
        users should remember that very small particles may require more
        careful parameter tuning.

        Particle Radius and Detection Sensitivity

        The particle radius is one of the most biologically meaningful
        parameters because it defines the expected scale of the target
        particle. A good practical estimate usually corresponds to roughly
        half the particle diameter measured in pixels.

        If the radius is set too small, extended complexes may be only
        partially represented, often increasing false positives or causing
        unstable center placement. If the radius is too large, nearby
        contaminants or neighboring particles may be merged into the
        prediction signal.

        The extraction threshold controls the confidence level required for
        accepting predicted particles. Lower thresholds typically recover
        more candidates, increasing sensitivity but also introducing more
        false positives. Higher thresholds provide more conservative
        coordinate sets, often preferred when preparing cleaner inputs for
        high-resolution refinement.

        In biological practice, the best threshold depends strongly on the
        purpose of the experiment. Initial exploratory rounds often benefit
        from permissive thresholds, whereas final production datasets
        generally require more stringent settings.

        Streaming and Large-Scale Processing

        The protocol is designed to support streaming and batch-oriented
        processing of large micrograph collections. This is particularly
        relevant in modern cryo-EM facilities where data may be acquired
        continuously during microscope sessions.

        From a biological workflow perspective, streaming enables early
        quality assessment. Users can quickly determine whether particles
        are visible, whether ice quality is acceptable, and whether the
        selected model is appropriate, often before the full acquisition
        session is complete.

        Parallel processing also makes the protocol suitable for large
        screening projects, ligand campaigns, or heterogeneous datasets
        collected under multiple conditions.

        Outputs and Interpretation

        The protocol produces a set of particle coordinates associated with
        the processed micrographs. These coordinates define the candidate
        particle centers and can be used directly for extraction and
        downstream classification.

        The resulting box size is determined so that extracted particles
        preserve enough surrounding context for later processing. From a
        biological standpoint, this is important because excessively small
        boxes may truncate flexible regions or peripheral domains, while
        overly large boxes may introduce unnecessary background noise.

        Users should always visually inspect a representative subset of the
        coordinates overlaid on micrographs. Even high-performing neural
        network pickers can be biased by contamination, carbon edges,
        crystalline ice, or preferred orientations that mimic true
        particles.

        Practical Recommendations

        For most biological applications, a good starting strategy is to
        begin with a pretrained general model and visually inspect the
        resulting picks. If particle localization appears systematically
        biased or incomplete, using a project-specific trained model often
        provides the largest improvement.

        Denoising can be especially helpful for small proteins, low-dose
        data, or challenging vitrification conditions, but it should be
        introduced conservatively. The particle radius should reflect the
        expected biological particle size as closely as possible, while the
        threshold should be adjusted depending on whether the goal is broad
        discovery or clean production picking.

        Final Perspective

        For cryo-EM users, automated particle picking is more than a
        convenience—it is one of the earliest biological filtering steps
        in the entire single-particle workflow. The quality of the selected
        coordinates strongly influences extraction, classification,
        reconstruction, and ultimately structural interpretation. Careful
        choice of model, thoughtful preprocessing, and biologically
        informed parameter tuning are therefore essential for reliable
        downstream results.
    """
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
        self.waitForCoordsFile(pickingFileName)

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

  def waitForCoordsFile(self, coordsFile, cMax=5):
      c = 0
      while (not os.path.exists(coordsFile) or os.path.getsize(coordsFile) == 0) and c < cMax:
          c += 1
          time.sleep(c)


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