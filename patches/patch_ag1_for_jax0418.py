from __future__ import annotations

import sys
from pathlib import Path


JAX_SHIM = """import sys
import types

# Compatibility for running the archived AG1 dependency stack with JAX 0.4.18.
try:
  import jax  # pylint: disable=g-import-not-at-top
  if not hasattr(jax, 'ShapedArray') and hasattr(jax, 'core'):
    jax.ShapedArray = jax.core.ShapedArray
  if 'jax.experimental.gda_serialization.serialization' not in sys.modules:
    gda_pkg = types.ModuleType('jax.experimental.gda_serialization')
    gda_ser = types.ModuleType('jax.experimental.gda_serialization.serialization')

    def _get_tensorstore_spec(path):
      return {'driver': 'file', 'path': path}

    class _GlobalAsyncCheckpointManager:
      def serialize(self, *args, **kwargs):
        raise NotImplementedError('GDA checkpointing is not used by this AG1 worker.')

      def deserialize(self, *args, **kwargs):
        raise NotImplementedError('GDA checkpointing is not used by this AG1 worker.')

    gda_ser.get_tensorstore_spec = _get_tensorstore_spec
    gda_ser.GlobalAsyncCheckpointManager = _GlobalAsyncCheckpointManager
    sys.modules['jax.experimental.gda_serialization'] = gda_pkg
    sys.modules['jax.experimental.gda_serialization.serialization'] = gda_ser
  if 'jax.experimental.global_device_array' not in sys.modules:
    gda_mod = types.ModuleType('jax.experimental.global_device_array')

    class _GlobalDeviceArray:
      pass

    gda_mod.GlobalDeviceArray = _GlobalDeviceArray
    sys.modules['jax.experimental.global_device_array'] = gda_mod
except Exception:  # pragma: no cover - best-effort compatibility shim.
  pass
"""


def replace_required(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def replace_optional(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")


def patch_alphageometry(repo: Path) -> None:
    path = repo / "alphageometry.py"
    text = path.read_text(encoding="utf-8")
    if "# Compatibility for running the archived AG1 dependency stack" not in text:
        text = text.replace("import traceback\n\n", "import traceback\n")
        marker = "import graph as gh\n"
        if marker not in text:
            raise RuntimeError("Could not find AG1 import marker in alphageometry.py")
        text = text.replace(marker, marker + JAX_SHIM)
    text = text.replace("with open(out_file, 'w') as f:", "with open(out_file, 'w', encoding='utf-8') as f:")
    path.write_text(text, encoding="utf-8")


def patch_float32(repo: Path) -> None:
    for rel in ["beam_search.py", "lm_inference.py", "models.py"]:
        path = repo / rel
        replace_optional(path, "dtype=jnp.bfloat16", "dtype=jnp.float32")
        replace_optional(path, "dtype=np.bfloat16", "dtype=np.float32")

    for rel in ["beam_search.py", "lm_inference.py", "models.py"]:
        text = (repo / rel).read_text(encoding="utf-8")
        if "bfloat16" in text:
            raise RuntimeError(f"Remaining bfloat16 reference in {rel}")


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (repo / "alphageometry.py").exists():
        raise FileNotFoundError(repo / "alphageometry.py")
    patch_alphageometry(repo)
    patch_float32(repo)
    print(f"Patched AG1 for JAX 0.4.18 at {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
