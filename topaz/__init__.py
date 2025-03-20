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

import pwem
import pyworkflow.utils as pwutils
import pyworkflow as pw

from .constants import *


__version__ = '3.1.0'
_references = ['Bepler2018']
_logo = "topaz_logo.jpeg"


class Plugin(pwem.Plugin):
    _supportedVersions = VERSIONS
    _url = "https://github.com/scipion-em/scipion-em-topaz"

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(TOPAZ_ENV_ACTIVATION, DEFAULT_ACTIVATION_CMD)

    @classmethod
    def getTopazEnvActivation(cls):
        """ Remove the scipion home and activate the conda topaz environment. """
        activation = cls.getVar(TOPAZ_ENV_ACTIVATION)
        scipionHome = pw.Config.SCIPION_HOME + os.path.sep

        return activation.replace(scipionHome, "", 1)

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch topaz. """
        environ = pwutils.Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']
        return environ

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = ['wget']
        if not condaActivationCmd:
            neededProgs.append('conda')

        return neededProgs

    @classmethod
    def defineBinaries(cls, env):

        cls.addTopazPackage(env, TOPAZ_DEFAULT_VER_NUM,
                            default=True)

    @classmethod
    def addTopazPackage(cls, env, version, default=False):
        TOPAZ_INSTALLED = 'topaz_%s_installed' % version
        ENV_NAME = getTopazEnvName(version)
        # try to get CONDA activation command
        installationCmd = cls.getCondaActivationCmd()

        # Create the environment
        installationCmd += 'conda create -y -n %s python=3.10 &&'\
                           % ENV_NAME

        # Activate the new environment
        installationCmd += 'conda activate %s &&' % ENV_NAME

        cudaVersion = cls.getVersionFromPath(pwem.Config.CUDA_LIB, pattern="cuda",
                                             default="11.6")

        # toolkitVersion = "10.2" if cudaVersion.major == 10 else "11.3"
        # Install downloaded code
        installationCmd += 'conda install -y topaz=%s fsspec pytorch-cuda=%s '\
                           '-c tbepler -c  pytorch -c nvidia&&' % (version, cudaVersion)

        # Flag installation finished
        installationCmd += 'touch %s' % TOPAZ_INSTALLED

        topaz_commands = [(installationCmd, TOPAZ_INSTALLED)]

        envPath = os.environ.get('PATH', "")
        # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None
        env.addPackage('topaz', version=version,
                       tar='void.tgz',
                       commands=topaz_commands,
                       neededProgs=cls.getDependencies(),
                       default=default,
                       vars=installEnvVars)

    @classmethod
    def runTopaz(cls, protocol, program, args, cwd=None):
        """ Run Topaz command from a given protocol. """
        fullProgram = '%s %s && %s' % (cls.getCondaActivationCmd(),
                                       cls.getTopazEnvActivation(), program)
        protocol.runJob(fullProgram, args, env=cls.getEnviron(), cwd=cwd)
