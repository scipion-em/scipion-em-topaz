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


def getTopazEnvName(version):
    return "topaz-%s" % version

V0_2_5 = "0.2.5"
V0_2_4 = "0.2.4"
V0_2_3 = "0.2.3"
V0_3_7 = "0.3.7"
VERSIONS = [V0_2_3, V0_2_4, V0_2_5, V0_3_7]
TOPAZ_DEFAULT_VER_NUM = V0_3_7

DEFAULT_ENV_NAME = getTopazEnvName(TOPAZ_DEFAULT_VER_NUM)
DEFAULT_ACTIVATION_CMD = 'conda activate ' + DEFAULT_ENV_NAME
TOPAZ_ENV_ACTIVATION = 'TOPAZ_ENV_ACTIVATION'

# Topaz supported input formats for micrographs
TOPAZ_SUPPORTED_FORMATS = [".mrc", ".tiff", ".png"]
