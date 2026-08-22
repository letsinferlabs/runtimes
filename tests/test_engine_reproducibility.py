from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEEPSEEK = ROOT / "dwarfstar--antirez--deepseek-v4-gguf--dgx-spark"


class EngineReproducibilityTests(unittest.TestCase):
    def test_dwarfstar_uses_a_distinct_stable_nvcc_seed_per_target(self) -> None:
        makefile = (DEEPSEEK / "engine/Makefile").read_text(encoding="utf-8")
        dockerfile = (DEEPSEEK / "image/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("--frandom-seed=$(subst /,_,$@)", makefile)
        self.assertIn("--objdir-as-tempdir", makefile)
        nvccflags = next(
            line for line in makefile.splitlines() if line.startswith("NVCCFLAGS ?=")
        )
        self.assertNotIn(" -g ", nvccflags)
        self.assertNotIn("-lineinfo", nvccflags)
        self.assertNotIn('NVCC_EXTRA_FLAGS="--frandom-seed=', dockerfile)


if __name__ == "__main__":
    unittest.main()
