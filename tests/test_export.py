"""deploy/export_policy.py: export, --verify, and the overwrite guards.

Uses a synthetic checkpoint with random actor weights, so it needs no trained
model. If DAB_V6_CKPT points at the dab's v6 model_20000.pt (not in this
repository), the shipped motions/dab/policy/policy.pt is also verified against it.
"""

import os
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy"))
import export_policy as ep  # noqa: E402

SCRIPT = os.path.join(ROOT, "deploy", "export_policy.py")
TEMPLATE = os.path.join(ROOT, "deploy", "policy_template", "config.yaml")


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def fake_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    actor = ep.ASAPActor()
    state = {k: v.clone() for k, v in actor.state_dict().items()}
    state["std"] = torch.ones(23)  # non-actor keys in the dict are ignored
    torch.save({"actor_model_state_dict": state, "iter": 0}, path)
    return state


def export(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)


class Export(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.ckpt = os.path.join(self.dir, "model_0.pt")
        self.state = fake_checkpoint(self.ckpt)
        self.out = os.path.join(self.dir, "policy", "policy.pt")

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_and_verify_pass(self):
        res = export("--ckpt", self.ckpt, "--out", self.out, "--verify", "--n", "16")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("VERIFY PASSED", res.stdout)
        policy = torch.jit.load(self.out)
        y = policy(torch.zeros(3, ep.RLSAR_OBS_DIM))
        self.assertEqual(tuple(y.shape), (3, 29))
        self.assertEqual(float(y[:, [19, 20, 21, 26, 27, 28]].abs().max()), 0.0)

    def test_verify_rejects_a_different_checkpoint(self):
        self.assertEqual(export("--ckpt", self.ckpt, "--out", self.out).returncode, 0)
        other = {k: (v + 1e-2 if k.startswith("actor_module.") else v) for k, v in self.state.items()}
        other = {k: v for k, v in other.items() if k.startswith("actor_module.")}
        self.assertFalse(ep.verify(self.out, other, 8, 0, 1e-5))

    def test_existing_policy_and_config_need_force(self):
        self.assertEqual(export("--ckpt", self.ckpt, "--out", self.out, "--config", TEMPLATE).returncode, 0)
        cfg = os.path.join(self.dir, "policy", "config.yaml")
        with open(cfg, "a") as f:
            f.write("# edited by hand\n")
        edited = read(cfg)

        res = export("--ckpt", self.ckpt, "--out", self.out, "--config", TEMPLATE)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(read(cfg), edited, "config.yaml must be kept without --force")

        os.remove(self.out)  # policy.pt gone, config.yaml still there
        res = export("--ckpt", self.ckpt, "--out", self.out, "--config", TEMPLATE)
        self.assertNotEqual(res.returncode, 0)
        self.assertEqual(read(cfg), edited)

        res = export("--ckpt", self.ckpt, "--out", self.out, "--config", TEMPLATE, "--force")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(read(cfg), read(TEMPLATE))

    @unittest.skipUnless(os.environ.get("DAB_V6_CKPT") and os.path.exists(os.environ.get("DAB_V6_CKPT", "")),
                         "DAB_V6_CKPT (the dab v6 model_20000.pt) is not in this repository")
    def test_shipped_dab_policy_matches_v6_checkpoint(self):
        shipped = os.path.join(ROOT, "motions", "dab", "policy", "policy.pt")
        self.assertTrue(ep.verify(shipped, ep.load_actor_state(os.environ["DAB_V6_CKPT"]), 64, 0, 1e-5))


if __name__ == "__main__":
    unittest.main()
