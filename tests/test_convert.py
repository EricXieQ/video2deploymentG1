"""capture/convert_tram_to_asap.py on a synthetic TRAM result.

The real regression check (re-converting a TRAM result and comparing the npz
byte for byte) needs TRAM output that is not in this repository; set
TRAM_RESULTS and CONVERT_EXPECTED_NPZ_DIR to run it. This test checks the
conversion's documented behaviour on made-up input.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "capture", "convert_tram_to_asap.py")


def convert(*args):
    env = {k: v for k, v in os.environ.items() if k not in ("ASAP_ROOT", "TRAM_ROOT")}
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, env=env)


class Convert(unittest.TestCase):
    def test_synthetic_track(self):
        n = 12
        with tempfile.TemporaryDirectory() as tmp:
            hps = os.path.join(tmp, "results", "clip", "hps")
            os.makedirs(hps)
            rot = np.tile(np.eye(3, dtype=np.float32), (n, 24, 1, 1))
            trans = np.zeros((n, 1, 3), dtype=np.float32)
            trans[:, 0, 0] = 0.3                                  # camera X (sideways)
            trans[:, 0, 1] = -np.linspace(0.0, 0.1, n)            # camera Y is down: body rises
            trans[:, 0, 2] = 4.0                                  # depth
            shape = np.arange(10, dtype=np.float32)[None].repeat(n, 0)
            np.save(os.path.join(hps, "hps_track_0.npy"),
                    {"pred_rotmat": rot, "pred_shape": shape, "pred_trans": trans}, allow_pickle=True)

            out = os.path.join(tmp, "out")
            res = convert("clip", "--fps", "60", "--smooth", "0",
                          "--tram-results", os.path.join(tmp, "results"), "--out-dir", out)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            d = np.load(os.path.join(out, "clip_amass.npz"))

            self.assertEqual(d["poses"].shape, (n, 72))
            self.assertEqual(int(d["mocap_framerate"]), 60)
            self.assertEqual(str(d["gender"]), "neutral")
            np.testing.assert_allclose(d["betas"][:10], np.arange(10))
            np.testing.assert_array_equal(d["betas"][10:], 0)
            np.testing.assert_array_equal(d["trans"][:, :2], 0)   # X and Y travel removed
            # camera Y-down becomes world Z-up: height follows -camera Y
            np.testing.assert_allclose(d["trans"][:, 2], np.linspace(0.0, 0.1, n), atol=1e-6)
            # root turned 90 deg about X (camera -> Z-up); other joints untouched (identity)
            np.testing.assert_allclose(d["poses"][:, :3], np.tile([-np.pi / 2, 0, 0], (n, 1)), atol=1e-5)
            np.testing.assert_array_equal(d["poses"][:, 3:], 0)

    def test_missing_out_dir_is_an_error(self):
        res = convert("clip", "--tram-results", "/nonexistent")
        self.assertNotEqual(res.returncode, 0)

    @unittest.skipUnless(os.environ.get("TRAM_RESULTS") and os.environ.get("CONVERT_EXPECTED_NPZ_DIR"),
                         "TRAM results are not in this repository")
    def test_reconvert_matches_existing_npz(self):
        exp = os.environ["CONVERT_EXPECTED_NPZ_DIR"]
        with tempfile.TemporaryDirectory() as tmp:
            for name in [f[:-len("_amass.npz")] for f in sorted(os.listdir(exp)) if f.endswith("_amass.npz")]:
                if not os.path.isdir(os.path.join(os.environ["TRAM_RESULTS"], name)):
                    continue
                res = convert(name, "--tram-results", os.environ["TRAM_RESULTS"], "--out-dir", tmp)
                self.assertEqual(res.returncode, 0, res.stderr)
                a, b = np.load(os.path.join(tmp, f"{name}_amass.npz")), np.load(os.path.join(exp, f"{name}_amass.npz"))
                for k in b.files:
                    np.testing.assert_array_equal(a[k], b[k], err_msg=f"{name}:{k}")


if __name__ == "__main__":
    unittest.main()
