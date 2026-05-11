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
from pwem.protocols import EMProtocol
import pyworkflow.protocol.params as params
import pyworkflow.protocol.constants as cons

class ProtTopazBase(EMProtocol):
    """
    Provides a common framework for Topaz-based cryo-EM preprocessing workflows, with emphasis on
    micrograph conditioning before particle detection or model training. Its purpose is to prepare
    raw micrographs so that downstream particle-picking and classification tasks operate on cleaner,
    more standardized image data.

    AI Generated:

    Topaz Preprocessing Base (ProtTopazBase) — User Manual
        Overview

        The Topaz Preprocessing Base protocol serves as the shared preprocessing foundation for
        workflows that depend on Topaz-based particle analysis. In cryo-EM practice, the quality of
        micrograph preparation strongly influences the quality of all later stages, particularly
        particle detection, neural-network training, and automated picking.

        The protocol is intended to standardize micrographs before analysis so that variations in
        noise level, image scale, and acquisition conditions have less impact on downstream
        interpretation. For biological users, this means that subtle particle features can become
        easier to recognize and more consistent across datasets.

        Denoising and Signal Preservation

        One of the central roles of the protocol is optional denoising. Cryo-EM micrographs often
        contain substantial high-frequency noise, and this can obscure weak particle signal,
        especially in challenging datasets involving small proteins, flexible complexes, or low-dose
        acquisitions.

        Denoising is designed to improve the visibility of biologically meaningful signal while
        preserving structural features that remain important for particle recognition. In practical
        workflows, this can improve the robustness of automated picking and reduce the number of
        false detections arising from noise or contamination.

        The protocol supports multiple denoising strategies because no single model is universally
        optimal. Some datasets benefit from stronger noise suppression, while others require a more
        conservative treatment to preserve weak but biologically relevant features.

        Large Micrographs and Patch-Based Processing

        Modern cryo-EM datasets frequently contain very large micrographs, especially when collected
        on high-resolution direct electron detectors. Processing these images as a whole can be
        computationally demanding, particularly on GPU-limited systems.

        For this reason, the protocol accommodates patch-based preprocessing. From a practical
        perspective, this allows large datasets to remain tractable without fundamentally changing
        the biological purpose of the workflow. The goal remains the same: preserve particle signal
        while maintaining computational feasibility.

        Downsampling and Scale Normalization

        In addition to denoising, the protocol performs image scaling so that downstream Topaz
        analyses operate at a biologically useful and computationally efficient sampling level.

        In cryo-EM particle picking, exact atomic detail is usually not necessary during the initial
        detection stage. Instead, what matters most is preserving particle-scale morphological
        information while reducing unnecessary computational burden.

        Proper scaling therefore helps balance biological interpretability and processing speed. It
        is particularly valuable in facility-scale pipelines or when screening large datasets where
        throughput is an important practical concern.

        Role in Particle Picking Workflows

        This protocol is not intended as a final biological interpretation step. Rather, it acts as
        an enabling stage for later analyses. Well-preprocessed micrographs tend to produce more
        stable neural-network behavior, more reproducible particle detection, and better initial
        particle sets for downstream classification and reconstruction.

        For biological users, the practical consequence is often improved dataset quality at an
        early stage of processing. Cleaner initial particle sets can reduce manual curation and
        improve the reliability of later structural results.

        GPU-Aware Processing Environment

        The protocol is designed for modern accelerated cryo-EM workflows and can take advantage of
        GPU resources during computationally intensive preprocessing tasks.

        In practical terms, this makes the protocol suitable both for exploratory laboratory use and
        for larger automated processing pipelines. GPU acceleration does not change the biological
        interpretation of the results, but it can make iterative optimization of preprocessing
        parameters far more practical.

        Model Continuity in Training Workflows

        In addition to preprocessing, the protocol provides continuity across Topaz learning
        workflows by maintaining access to the most relevant trained model state.

        This is particularly useful when preprocessing is embedded within iterative training,
        retraining, or dataset expansion workflows. Biological users benefit because model
        refinement can continue smoothly as datasets grow or as picking strategies are adjusted.

        Practical Recommendations

        For most cryo-EM datasets, it is often useful to begin with conservative preprocessing and
        inspect the resulting micrographs visually. Excessive denoising may suppress weak but
        biologically meaningful signal, especially in small particles or flexible complexes.

        Moderate downsampling usually provides a good starting point for automated picking.
        Particularly noisy datasets often benefit from denoising first, whereas very clean
        micrographs may require only scale normalization.

        When testing new datasets, it is generally advisable to compare preprocessing settings
        against a small representative subset before launching large-scale picking or training.

        Final Perspective

        In practical cryo-EM analysis, preprocessing is not merely a computational convenience but
        a biologically important preparation stage. The clarity of particle signal at this stage can
        strongly affect all downstream interpretations.

        By combining denoising, scale normalization, and workflow continuity into a common Topaz
        foundation, this protocol helps transform raw micrographs into more reliable inputs for
        automated particle analysis and subsequent structural investigation.
    """
  def __init__(self, **args):
    EMProtocol.__init__(self, **args)

  def _definePreprocessParams(self, form):
    form.addSection('Pre-process')
    group = form.addGroup('Denoise')
    group.addParam('doDenoise', params.BooleanParam, default=False,
                   label="Denoise micrographs?")
    group.addParam('modelDenoise', params.EnumParam, default=0,
                   condition='doDenoise',
                   choices=['unet', 'unet-small', 'fcnn', 'affineresnet8'],
                   label='Model',
                   help='Denoising model to use on micrographs.')
    group.addParam('patchSize', params.IntParam, default=-1,
                   label='Patch Size', condition='doDenoise',
                   help='Process each micrograph in patches of this size.\n'
                        'This is useful when using GPU processing and the micrographs '
                        'are too large to be denoised in one shot on your GPU. '
                        'By default (<0), it is not used')
    group.addParam('denoiseExtra', params.StringParam, default='',
                   expertLevel=cons.LEVEL_ADVANCED, condition='doDenoise',
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

    form.addHidden(params.GPU_LIST, params.StringParam, default='0',
                   expertLevel=cons.LEVEL_ADVANCED,
                   label="Choose GPU IDs",
                   help="GPU may have several cores. Set it to zero"
                        " if you do not know what we are talking about."
                        " First core index is 0, second 1 and so on.")


  #UTILS for preprocess steps
  def getDenoiseArgs(self, inputDir, outDir):
      args = ' %s/*.mrc -o %s/' % (inputDir, outDir)
      args += ' --model %s' % self.getEnumText('modelDenoise')
      args += ' --device %(GPU)s'  # Add GPU that will be set by the executor
      if self.patchSize.get() > 0:
        args += ' --patch-size %s' % self.patchSize.get()

      if self.denoiseExtra.hasValue():
        args += ' ' + self.denoiseExtra.get()
      else:
        args += ' --normalize'

      return args

  def getPreprocessArgs(self, inputDir, outDir):
    args = " %s/*.mrc -o %s/" % (inputDir, outDir)
    args += " --scale %d " % self.scale.get()
    args += ' --num-workers %d' % self.numberOfThreads
    args += ' --device %(GPU)s'  # Add GPU that will be set by the executor

    if self.preExtra.hasValue():
      args += ' ' + self.preExtra.get()

    return args

  def getOutputModelPath(self):
    return self.MODEL

  def getLastEpochModel(self, modelsDir, ext='.sav'):
    '''Return the last trained model, in alphabetic order (last trained epoch) in modelsDir'''
    modelFn = 'model.sav'
    for file in os.listdir(modelsDir):
      if ext in file:
        modelFn = file
    return os.path.join(modelsDir, modelFn)





