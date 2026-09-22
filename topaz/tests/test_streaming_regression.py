# ***************************************************************************
# *
# * Regression tests for Topaz streaming/Continue behaviour.
# *
# ***************************************************************************

import os
import tempfile
import unittest

from topaz.protocols.protocol_topaz_picking import TopazProtPicking


class _Mic:
    def __init__(self, obj_id):
        self._obj_id = obj_id

    def strId(self):
        return str(self._obj_id)


class _PickingHarness:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def _getTmpPath(self):
        return self.tmp_dir


class TestTopazStreamingRegression(unittest.TestCase):
    def testContinueFindsOriginalBatchForPendingSubset(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "micrographs1-32"))

            # Simulate a resumed batch where coordinates for mics 1-10 were
            # already persisted before the crash. The core therefore asks
            # Topaz only for the still-pending subset 11-32.
            pending_mics = [_Mic(i) for i in range(11, 33)]
            protocol = _PickingHarness(tmp)

            batches = TopazProtPicking.getPickingMinMax(
                protocol,
                pending_mics,
            )

            self.assertEqual(
                batches,
                [("1", "32")],
                "Continue must recover the original Topaz batch whose range "
                "contains the pending subset, otherwise its coordinate file "
                "cannot be re-read after a partial persistence crash.",
            )


class _Value:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _OutputCoords:
    def __init__(self):
        self.box_size = None

    def getMicrographs(self):
        return object()

    def setBoxSize(self, value):
        self.box_size = value


class _ReadHarness:
    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        self.scale = _Value(1)
        self.boxSize = _Value(-1)
        self.radius = _Value(8)

    def getPickingMinMax(self, mic_list):
        return [("1", "32")]

    def _getFileName(self, key, **kwargs):
        return os.path.join(
            self.tmp_dir,
            "topaz_coordinates%s-%s.txt" % (kwargs["min"], kwargs["max"]),
        )

    def waitForCoordsFile(self, coords_file):
        pass


def _testContinueReadsOnlyPendingMicrographs(self):
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        protocol = _ReadHarness(tmp)
        pending_mics = [_Mic(i) for i in range(11, 33)]
        output_coords = _OutputCoords()

        with patch(
            "topaz.protocols.protocol_topaz_picking.readSetOfCoordinates"
        ) as mocked_read:
            TopazProtPicking.readCoordsFromMics(
                protocol,
                tmp,
                pending_mics,
                output_coords,
            )

        picking_file = os.path.join(
            tmp,
            "topaz_coordinates1-32.txt",
        )
        mocked_read.assert_called_once_with(
            picking_file,
            pending_mics,
            output_coords,
            1,
        )


TestTopazStreamingRegression.testContinueReadsOnlyPendingMicrographs = (
    _testContinueReadsOnlyPendingMicrographs
)


class _ConvertibleMic:
    def __init__(self, obj_id):
        self._obj_id = obj_id

    def getObjId(self):
        return self._obj_id

    def clone(self):
        return _ConvertibleMic(self._obj_id)


class _CoordSet:
    def __init__(self):
        self.items = []

    def append(self, coord):
        self.items.append(coord.clone())


class _FakeScore:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class _FakeCoordinate:
    def __init__(self):
        self.mic_id = None
        self.position = None
        self._topazScore = None
        self.obj_id = None

    def setMicrograph(self, mic):
        self.mic_id = mic.getObjId()

    def setPosition(self, x, y):
        self.position = (x, y)

    def setObjId(self, obj_id):
        self.obj_id = obj_id

    def clone(self):
        cloned = _FakeCoordinate()
        cloned.mic_id = self.mic_id
        cloned.position = self.position
        cloned.obj_id = self.obj_id
        cloned._topazScore = _FakeScore()
        cloned._topazScore.value = self._topazScore.value
        return cloned


class _FakeCsvCoordinateList:
    def __init__(self, *args, **kwargs):
        self.rows = [
            ["1", "10", "20", "0.5"],
            ["11", "30", "40", "0.9"],
        ]

    def __iter__(self):
        return iter(self.rows)

    def close(self):
        pass


def _testCoordinateReaderSkipsAlreadyPersistedMicrographs(self):
    from unittest.mock import patch
    from topaz import convert

    coord_set = _CoordSet()
    pending_mics = [_ConvertibleMic(11)]

    with patch.object(convert, "CsvCoordinateList", _FakeCsvCoordinateList), \
         patch.object(convert, "Coordinate", _FakeCoordinate), \
         patch.object(convert, "Float", _FakeScore):
        convert.readSetOfCoordinates(
            "unused.txt",
            pending_mics,
            coord_set,
            1,
        )

    self.assertEqual(len(coord_set.items), 1)
    self.assertEqual(coord_set.items[0].mic_id, 11)
    self.assertEqual(coord_set.items[0].position, (30, 40))


TestTopazStreamingRegression.testCoordinateReaderSkipsAlreadyPersistedMicrographs = (
    _testCoordinateReaderSkipsAlreadyPersistedMicrographs
)


if __name__ == "__main__":
    unittest.main()
