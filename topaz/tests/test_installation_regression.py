# ***************************************************************************
# *
# * Regression tests for the Topaz installer.
# *
# ***************************************************************************

import unittest
from unittest.mock import patch

from topaz import Plugin


class _InstallEnv:
    def __init__(self):
        self.packages = []

    def addPackage(self, name, **kwargs):
        self.packages.append((name, kwargs))


class TestTopazInstallationRegression(unittest.TestCase):
    def testInstallerBuildsValidatedCompatibleCudaEnvironment(self):
        env = _InstallEnv()

        with patch.object(
            Plugin,
            "getCondaActivationCmd",
            return_value='eval "$(conda shell.bash hook)" && ',
        ), patch.object(
            Plugin,
            "getVersionFromPath",
            return_value="11.8",
        ), patch.object(
            Plugin,
            "getDependencies",
            return_value=["wget"],
        ):
            Plugin.addTopazPackage(env, "0.3.7", default=True)

        self.assertEqual(len(env.packages), 1)
        name, package = env.packages[0]
        self.assertEqual(name, "topaz")

        command, target = package["commands"][0]

        self.assertIn("--override-channels", command)
        self.assertIn("pytorch==2.3.1", command)
        self.assertIn("torchvision==0.18.1", command)
        self.assertIn("pytorch-cuda=11.8", command)
        self.assertIn("'mkl<2024.1'", command)
        self.assertIn("'setuptools<81'", command)

        validation = (
            'python -c "import torch, torchvision; '
            'assert torch.version.cuda is not None"'
        )
        self.assertIn(validation, command)
        self.assertIn("topaz --help", command)

        validation_pos = command.index(validation)
        help_pos = command.index("topaz --help")
        touch_pos = command.index("touch topaz_0.3.7_installed")

        self.assertLess(validation_pos, touch_pos)
        self.assertLess(help_pos, touch_pos)
        self.assertEqual(target, "topaz_0.3.7_installed")


if __name__ == "__main__":
    unittest.main()
