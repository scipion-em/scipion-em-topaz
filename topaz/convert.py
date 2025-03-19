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

import csv
import os

import pyworkflow.utils as pwutils
from pyworkflow.object import Float
from pwem.emlib.image import ImageHandler
from pwem.objects import Coordinate

from topaz import constants


class CsvImageList:
    """ Handler class to write a list of images as expected by topaz. """
    def __init__(self, filename, mode='r', **kwargs):
        self.__file = None

        if mode == 'r':
            self.__file = open(filename, 'r')
            self.__reader = csv.reader(self.__file, delimiter='\t')
        elif mode == 'w':
            self.__file = open(filename, 'w')
            self.__writer = csv.writer(self.__file, delimiter='\t')
            self.__writer.writerow(kwargs['columns'])

    def _addRow(self, *values):
        self.__writer.writerow(values)

    def close(self):
        self.__file.close()

    def __iter__(self):
        it = iter(self.__reader)
        # Just discard the first row
        self.__columns = next(it)
        return it


class CsvMicrographList(CsvImageList):
    """ Handler class to write a list of micrographs as expected by topaz. """
    def __init__(self, filename, mode='r'):
        CsvImageList.__init__(self, filename, mode,
                              columns=['image_name', 'path'])

    def addMic(self, micId, micPath):
        self._addRow('%06d' % micId, micPath)


class CsvCoordinateList(CsvImageList):
    """ Handler class to write a list of particles as expected by topaz. """
    def __init__(self, filename, mode='r', score=False):
        columns = ['image_name', 'x_coord', 'y_coord']
        if score:
            columns.append('score')

        CsvImageList.__init__(self, filename, mode, columns=columns)

    def addCoord(self, micId, x, y):
        self._addRow(micId2MicName(micId), x, y)


def micId2MicName(micId):
    return '%06d' % micId

def convertMicrographs(micList, micDir):
    """ Convert (or simply link) input micrographs into the given directory
    in a format that is compatible with Topaz.
    """
    ih = ImageHandler()
    ext = pwutils.getExt(micList[0].getFileName())

    def _convert(mic, newName):
        ih.convert(mic, os.path.join(micDir, newName))

    def _link(mic, newName):
        pwutils.createAbsLink(os.path.abspath(mic.getFileName()),
                              os.path.join(micDir, newName))

    if ext in constants.TOPAZ_SUPPORTED_FORMATS:
        func = _link
    else:
        func = _convert
        ext = '.mrc'

    for mic in micList:
        func(mic, getMicIdName(mic, suffix=ext))


def readSetOfCoordinates(coordinatesCsvFn, micSet, coordSet, scale):
    """ Read coordinates produced by Topaz.
    Coordinates are expected in a single csv file, with the following columns:
     first: image_name (mic id)
     second: x_coord
     third:  y_coord
     forth:  score
    """
    csv = CsvCoordinateList(coordinatesCsvFn, score=True)

    lastMicId = None
    coord = Coordinate()
    coord._topazScore = Float()

    micDict = {}
    # loop to generate a dictionary --> micBaseName : Micrograph
    for mic in micSet:
        micNew = mic.clone()
        micDict[mic.getObjId()] = micNew

    #loop the Topaz outputfile
    for row in csv:
        micId = int(row[0])
        if micId != lastMicId:
            mic = micDict[micId]
            if mic is None:
                print("Missing id: ", micId)
            else:
                coord.setMicrograph(mic)
                lastMicId = micId

        coord.setPosition(int(round(float(row[1])*scale)), int(round(float(row[2])*scale)))
        coord._topazScore.set(float(row[3]))
        coord.setObjId(None)
        coordSet.append(coord)

    csv.close()


def getMicIdName(mic, suffix=''):
    """ Return a name for the micrograph based on its IDs. """
    return '%d%s' % (mic.getObjId(), suffix)
