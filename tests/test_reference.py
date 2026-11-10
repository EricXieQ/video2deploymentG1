"""check_reference.py and ground_reference.py on the shipped dab reference motion.

The expected numbers are the ones measured when the repository was built
(motions/dab/README.md): lower ankle_roll link 0.045 to 0.097 m, root height
0.800 to 0.851 m. Needs ASAP_ROOT (see tests/run_tests.sh) for the robot model.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "capture"))
import reference_fk as rf  # noqa: E402

DAB_PKL = os.path.join(ROOT, "motions/dab/0-motions_raw_tairantestbed_smpl_wall_background_dab_amass.pkl")
ASAP_ROOT = os.environ.get("ASAP_ROOT")


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "capture", script), "--asap-root", ASAP_ROOT, *args],
                          capture_output=True, text=True)


@unittest.skipUnless(ASAP_ROOT and os.path.exists(os.path.join(ASAP_ROOT or "", rf.DEFAULT_MJCF)),
                     "ASAP_ROOT with the G1 robot model is not available")
class DabReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = rf.load_robot(os.path.join(ASAP_ROOT, rf.DEFAULT_MJCF))
        _, cls.motion = rf.load_motion(DAB_PKL)

    def test_known_floating_feet_and_root_height(self):
        lowest = rf.body_heights(self.robot, self.motion).min(axis=1)
        self.assertAlmostEqual(lowest.min(), 0.045, delta=0.002)
        self.assertAlmostEqual(lowest.max(), 0.097, delta=0.002)
        z = np.asarray(self.motion["root_trans_offset"])[:, 2]
        self.assertAlmostEqual(float(z.min()), 0.800, delta=0.001)
        self.assertAlmostEqual(float(z.max()), 0.851, delta=0.001)

    def test_known_pelvis_tilt(self):
        self.assertAlmostEqual(float(rf.pelvis_tilt_deg(self.motion).mean()), 22.0, delta=0.5)

    def test_check_reference_cli_warns_but_exits_zero_unless_strict(self):
        normal = run("check_reference.py", DAB_PKL)
        self.assertEqual(normal.returncode, 0, normal.stderr)
        strict = run("check_reference.py", "--strict", DAB_PKL)
        self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)

    def test_ground_reference_pins_lower_foot_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, os.path.basename(DAB_PKL))
            shutil.copyfile(DAB_PKL, src)
            before = read(src, "rb")

            res = run("ground_reference.py", src)
            self.assertEqual(res.returncode, 0, res.stderr)
            out = src.replace(".pkl", "_grounded.pkl")
            self.assertTrue(os.path.exists(out))
            self.assertEqual(read(src, "rb"), before, "input must not change")

            data = joblib.load(out)
            self.assertEqual(list(data), [os.path.basename(out)[:-4]], "motion key must match the file stem")
            grounded = next(iter(data.values()))
            lowest = rf.body_heights(self.robot, grounded).min(axis=1)
            self.assertLess(float(np.ptp(lowest)), 0.001)
            self.assertAlmostEqual(float(lowest.mean()), rf.FLOOR_ANKLE_ROLL_Z, delta=0.0005)
            # only the vertical root position changes
            np.testing.assert_array_equal(grounded["dof"], self.motion["dof"])
            np.testing.assert_array_equal(grounded["root_rot"], self.motion["root_rot"])
            np.testing.assert_array_equal(np.asarray(grounded["root_trans_offset"])[:, :2],
                                          np.asarray(self.motion["root_trans_offset"])[:, :2])

            again = run("ground_reference.py", src)
            self.assertNotEqual(again.returncode, 0, "an existing output must not be replaced without --force")

            strict = run("check_reference.py", out)
            self.assertEqual(strict.returncode, 0, strict.stderr)


if __name__ == "__main__":
    unittest.main()
