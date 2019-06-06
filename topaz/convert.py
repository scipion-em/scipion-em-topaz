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

import csv
import pyworkflow.utils as pwutils
import pyworkflow as pw


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
        self._addRow('%06d' % micId, x, y)


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
    coord = pw.em.Coordinate()
    coord._topazScore = pw.object.Float()

    micDict = {}
    # loop to generate a dictionary --> micBaseName : Micrograph
    for mic in micSet:
        micBaseName = pwutils.removeBaseExt(mic.getFileName())
        micNew = mic.clone()
        micDict[micBaseName] = micNew

    for row in csv:
        micBaseName = (row[0])
        if micBaseName != lastMicId:
            print("New mic: ", micBaseName)
            mic = micDict[micBaseName]
            if mic is None:
                print("Missing id: ", micBaseName)
            else:
                coord.setMicrograph(mic)
                lastMicId = micBaseName

        coord.setPosition(int(round(float(row[1])*scale)), int(round(float(row[2])*scale)))
        coord._topazScore.set(float(row[3]))
        coord.setObjId(None)
        coordSet.append(coord)

    return
