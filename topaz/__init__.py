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
import pyworkflow.em
import pyworkflow as pw
from .constants import CONDA_ACTIVATION_CMD, TOPAZ_ACTIVATION_CMD,DEFAULT_ACTIVATION_CMD, DEFAULT_ENV_NAME
import topaz

_references = ['Bepler2018']
_logo = "topaz_logo.jpeg"


class Plugin(pw.em.Plugin):
    _supportedVersions = []

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(TOPAZ_ACTIVATION_CMD, DEFAULT_ACTIVATION_CMD)

    @classmethod
    def getCondaActivationCmd(cls):
        condaActivationCmd = os.environ.get('CONDA_ACTIVATION_CMD', "")
        correctCondaActivationCmd = condaActivationCmd.replace(pw.Config.SCIPION_HOME + "/", "")
        if not correctCondaActivationCmd:
            print("WARNING!!: CONDA_ACTIVATION_CMD variable not defined. "
                   "Relying on conda being in the PATH")
        elif correctCondaActivationCmd[-1] != ";":
            correctCondaActivationCmd += ";"
        return correctCondaActivationCmd

    @classmethod
    def getTopazEnvActivation(cls):
        """ Remove the scipion home and activate the conda topaz environment. """
        topazActivationCmd = cls.getVar(TOPAZ_ACTIVATION_CMD)
        correctCommand = topazActivationCmd.replace(pw.Config.SCIPION_HOME + "/", "")

        return cls.getCondaActivationCmd() + " " + correctCommand

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch topaz. """
        environ = pw.utils.Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']
        return environ

    @classmethod
    def defineBinaries(cls, env):

        TOPAZ_INSTALLED = 'topaz_installed'

        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs = ['conda']

        installationCmd = '%s conda create -y -n %s;conda activate %s; conda install -y topaz cudatoolkit=9.2 -c tbepler -c pytorch; touch %s' % \
                           (condaActivationCmd, DEFAULT_ENV_NAME, DEFAULT_ENV_NAME, TOPAZ_INSTALLED)
        topaz_commands = [(installationCmd, TOPAZ_INSTALLED)]

        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None
        env.addPackage('topaz', version='0.2.1',
                       tar='void.tgz',
                       commands=topaz_commands,
                       neededProgs=neededProgs,
                       default=True,
                       vars=installEnvVars)



pw.em.Domain.registerPlugin(__name__)
