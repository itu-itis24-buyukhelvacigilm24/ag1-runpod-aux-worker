from __future__ import annotations

import importlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


BUILTIN_PROBLEMS: dict[str, str] = {
    "orthocenter": (
        "orthocenter\n"
        "a b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
    ),
}


class AG1AuxService:
    def __init__(self) -> None:
        self.ag1_repo = Path(os.environ.get("AG1_REPO", "/opt/alphageometry"))
        self.meliad_root = Path(os.environ.get("MELIAD_ROOT", "/opt/meliad"))
        self.ckpt_dir = Path(os.environ.get("AG1_CKPT_DIR", "/opt/ag_ckpt_vocab_hf"))
        self.ckpt_subdir = os.environ.get("AG1_CKPT_SUBDIR", "ag_ckpt_vocab")
        self.vocab_path = Path(
            os.environ.get("AG1_VOCAB_PATH", str(self.ckpt_dir / "geometry.757.model"))
        )
        self.hf_repo = os.environ.get(
            "AG1_HF_REPO", "abrahamabelboodala/ALPHAGEOMETRY_ag_ckpt_vocab"
        )
        self.default_beam_size = int(os.environ.get("AG1_BEAM_SIZE", "4"))
        self.default_batch_size = int(os.environ.get("AG1_BATCH_SIZE", "4"))
        self.min_batch_size = int(os.environ.get("AG1_MIN_BATCH_SIZE", "4"))
        self.default_sequence_length = int(os.environ.get("AG1_SEQUENCE_LENGTH", "128"))
        self.translation_retries = int(os.environ.get("AG1_TRANSLATION_RETRIES", "12"))

        self._ag = None
        self._model = None
        self._load_elapsed_sec: float | None = None
        self._model_batch_size: int | None = None
        self._model_sequence_length: int | None = None

    def healthcheck(self, load_model: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ok": True,
            "service": "ag1_aux_worker",
            "model_role": "AG1 public auxiliary-construction LM, not full AG2 Gemini LM",
            "public_parameter_count": "152M",
            "ag1_repo": str(self.ag1_repo),
            "meliad_root": str(self.meliad_root),
            "ckpt_dir": str(self.ckpt_dir),
            "model_loaded": self._model is not None,
        }
        if load_model:
            self.load()
            data["model_loaded"] = True
            data["load_elapsed_sec"] = self._load_elapsed_sec
        try:
            import jax  # pylint: disable=import-outside-toplevel

            data["jax_devices"] = [str(device) for device in jax.devices()]
            data["jax_default_backend"] = jax.default_backend()
        except Exception as exc:  # pragma: no cover - diagnostic only
            data["jax_error"] = repr(exc)
        return data

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_beam_size = int(payload.get("beam_size", self.default_beam_size))
        requested_batch_size = int(payload.get("batch_size", self.default_batch_size))
        requested_sequence_length = int(payload.get("sequence_length", self.default_sequence_length))
        # AG1/Meliad returns one candidate per configured batch slot. The
        # public AlphaGeometry proof-search layer also calls its queue width
        # "beam_size", but lm_inference.beam_decode() does not accept a
        # per-call beam argument. Keep batch at least as large as requested
        # beam or the worker silently cannot return enough candidates.
        effective_batch_size = max(
            requested_batch_size,
            requested_beam_size,
            self.min_batch_size,
        )
        self.load(
            batch_size=effective_batch_size,
            sequence_length=requested_sequence_length,
        )
        problems = self._resolve_problems(payload)
        started = time.perf_counter()
        results = [
            self._probe_one(
                problem_id=problem_id,
                raw_problem=raw_problem,
                beam_size=requested_beam_size,
            )
            for problem_id, raw_problem in problems
        ]
        candidate_capacity = self._model_batch_size or 0
        return {
            "ok": True,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "load_elapsed_sec": self._load_elapsed_sec,
            "model_role": "AG1 public auxiliary-construction LM, not full AG2 Gemini LM",
            "public_parameter_count": "152M",
            "requested_beam_size": requested_beam_size,
            "requested_batch_size": requested_batch_size,
            "effective_batch_size": effective_batch_size,
            "model_batch_size": self._model_batch_size,
            "model_sequence_length": self._model_sequence_length,
            "candidate_capacity": candidate_capacity,
            "beam_truncated": requested_beam_size > candidate_capacity,
            "count": len(results),
            "results": results,
        }

    def load(self, batch_size: int | None = None, sequence_length: int | None = None) -> None:
        if self._model is not None:
            if batch_size and self._model_batch_size and batch_size > self._model_batch_size:
                raise RuntimeError(
                    "AG1 model is already loaded with batch_size="
                    f"{self._model_batch_size}, but this request needs batch_size={batch_size}. "
                    "Restart the worker or set AG1_MIN_BATCH_SIZE high enough before first load."
                )
            return

        started = time.perf_counter()
        self._check_paths()
        self._ensure_checkpoint()

        sys.path.insert(0, str(self.ag1_repo))
        sys.path.insert(0, str(self.meliad_root))
        os.chdir(self.ag1_repo)

        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if os.environ.get("AG1_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self._install_jax_legacy_shims()
        ag = importlib.import_module("alphageometry")
        self._configure_absl(
            batch_size=batch_size or self.default_batch_size,
            sequence_length=sequence_length or self.default_sequence_length,
        )
        ag.DEFINITIONS = ag.pr.Definition.from_txt_file(
            str(self.ag1_repo / "defs.txt"), to_dict=True
        )
        ag.RULES = ag.pr.Theorem.from_txt_file(str(self.ag1_repo / "rules.txt"), to_dict=True)
        self._ag = ag
        self._model = ag.get_lm(str(self.ckpt_dir), str(self.vocab_path))
        self._model_batch_size = int(getattr(self._model, "batch_size", batch_size or self.default_batch_size))
        self._model_sequence_length = sequence_length or self.default_sequence_length
        self._load_elapsed_sec = round(time.perf_counter() - started, 3)

    def _check_paths(self) -> None:
        missing = [
            label
            for label, path in {
                "AG1_REPO": self.ag1_repo,
                "MELIAD_ROOT": self.meliad_root,
            }.items()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing required paths: {missing}")

    def _ensure_checkpoint(self) -> None:
        required = [
            self.ckpt_dir / "checkpoint_10999999",
            self.ckpt_dir / "geometry.757.model",
            self.ckpt_dir / "geometry.757.vocab",
        ]
        if all(path.exists() for path in required):
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.hf_repo,
            local_dir=str(self.ckpt_dir),
            local_dir_use_symlinks=False,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing and self.ckpt_subdir:
            nested = self.ckpt_dir / self.ckpt_subdir
            nested_required = [
                nested / "checkpoint_10999999",
                nested / "geometry.757.model",
                nested / "geometry.757.vocab",
            ]
            if all(path.exists() for path in nested_required):
                self.ckpt_dir = nested
                self.vocab_path = Path(
                    os.environ.get("AG1_VOCAB_PATH", str(nested / "geometry.757.model"))
                )
                return
            missing = [str(path) for path in nested_required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint download did not create: {missing}")

    def _configure_absl(self, batch_size: int, sequence_length: int) -> None:
        from absl import flags  # pylint: disable=import-outside-toplevel

        if flags.FLAGS.is_parsed():
            return
        flags.FLAGS(
            [
                "ag1_runpod_worker",
                f"--gin_search_paths={self.meliad_root / 'transformer' / 'configs'},{self.ag1_repo}",
                "--gin_file=base_htrans.gin",
                "--gin_file=options/enable_scan.gin",
                "--gin_file=size/medium_150M.gin",
                "--gin_file=options/positions_t5.gin",
                "--gin_file=options/lr_cosine_decay.gin",
                "--gin_file=options/seq_1024_nocache.gin",
                "--gin_file=geometry_150M_generate.gin",
                "--gin_param=DecoderOnlyLanguageModelGenerate.output_token_losses=True",
                f"--gin_param=TransformerTaskConfig.batch_size={batch_size}",
                f"--gin_param=TransformerTaskConfig.sequence_length={sequence_length}",
                "--gin_param=Trainer.restore_state_variables=False",
            ]
        )

    @staticmethod
    def _install_jax_legacy_shims() -> None:
        try:
            import jax  # pylint: disable=import-outside-toplevel

            if not hasattr(jax, "ShapedArray") and hasattr(jax, "core"):
                jax.ShapedArray = jax.core.ShapedArray

            if "jax.experimental.gda_serialization.serialization" not in sys.modules:
                gda_pkg = types.ModuleType("jax.experimental.gda_serialization")
                gda_ser = types.ModuleType("jax.experimental.gda_serialization.serialization")

                def _get_tensorstore_spec(path):
                    return {"driver": "file", "path": path}

                class _GlobalAsyncCheckpointManager:
                    def serialize(self, *args, **kwargs):
                        raise NotImplementedError("GDA checkpointing is not used here.")

                    def deserialize(self, *args, **kwargs):
                        raise NotImplementedError("GDA checkpointing is not used here.")

                gda_ser.get_tensorstore_spec = _get_tensorstore_spec
                gda_ser.GlobalAsyncCheckpointManager = _GlobalAsyncCheckpointManager
                sys.modules["jax.experimental.gda_serialization"] = gda_pkg
                sys.modules["jax.experimental.gda_serialization.serialization"] = gda_ser

            if "jax.experimental.global_device_array" not in sys.modules:
                gda_mod = types.ModuleType("jax.experimental.global_device_array")

                class _GlobalDeviceArray:
                    pass

                gda_mod.GlobalDeviceArray = _GlobalDeviceArray
                sys.modules["jax.experimental.global_device_array"] = gda_mod
        except Exception:
            pass

    def _resolve_problems(self, payload: dict[str, Any]) -> list[tuple[str, str]]:
        if "problems" in payload:
            rows: list[tuple[str, str]] = []
            for idx, item in enumerate(payload["problems"]):
                if isinstance(item, str):
                    rows.append((self._problem_name(item, f"problem_{idx}"), item))
                else:
                    raw = str(item.get("problem") or item.get("text") or "")
                    rows.append((str(item.get("problem_id") or self._problem_name(raw, f"problem_{idx}")), raw))
            return rows

        if "problem" in payload:
            raw_problem = str(payload["problem"])
            return [(str(payload.get("problem_id") or self._problem_name(raw_problem, "custom")), raw_problem)]

        problem_id = str(payload.get("problem_id", "orthocenter"))
        if problem_id not in BUILTIN_PROBLEMS:
            raise KeyError(f"Unknown problem_id {problem_id!r}. Known: {sorted(BUILTIN_PROBLEMS)}")
        return [(problem_id, BUILTIN_PROBLEMS[problem_id])]

    @staticmethod
    def _problem_name(raw_problem: str, fallback: str) -> str:
        first = raw_problem.strip().splitlines()[0].strip() if raw_problem.strip() else ""
        return first or fallback

    def _probe_one(self, problem_id: str, raw_problem: str, beam_size: int) -> dict[str, Any]:
        if self._ag is None or self._model is None:
            raise RuntimeError("Model is not loaded.")

        started = time.perf_counter()
        ag = self._ag
        problem, graph, graph_attempts = self._build_problem_graph_robust(raw_problem)
        prompt = problem.setup_str_from_problem(ag.DEFINITIONS) + " {F1} x00"
        outputs = self._model.beam_decode(prompt, eos_tokens=[";"])
        candidate_capacity = len(outputs.get("seqs_str", []))
        truncated = beam_size > candidate_capacity

        rows: list[dict[str, Any]] = []
        for idx, (lm_out, score) in enumerate(zip(outputs["seqs_str"], outputs["scores"]), start=1):
            translation, translation_meta = self._translate_aux_robust(raw_problem, lm_out, graph)
            rows.append(
                {
                    "raw_rank": idx,
                    "lm_output": lm_out,
                    "score": float(score),
                    "translation": translation,
                    "valid": not translation.startswith("ERROR:"),
                    **translation_meta,
                }
            )
        # Preserve AG1/Meliad output order. The score is an internal sequence
        # value/loss and should not be treated as a stable cross-prompt ranking
        # signal after grammar translation.
        sorted_rows = list(rows)
        for idx, row in enumerate(sorted_rows, start=1):
            row["rank"] = idx

        return {
            "ok": True,
            "problem_id": problem_id,
            "raw_problem": raw_problem,
            "translated_problem": problem.txt(),
            "lm_prompt": prompt,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "beam_size_requested": beam_size,
            "candidate_capacity": candidate_capacity,
            "beam_truncated": truncated,
            "graph_build_attempts": graph_attempts,
            "translation_retries": self.translation_retries,
            "suggestions": sorted_rows[:beam_size],
            "suggestions_raw_order": rows,
        }

    @staticmethod
    def _is_transient_numeric_translation_error(text: str) -> bool:
        return any(
            marker in text
            for marker in [
                "PointTooCloseError",
                "PointTooFarError",
                "InvalidLineIntersectError",
            ]
        )

    def _build_problem_graph_robust(self, raw_problem: str):
        """Build AG1 graph with retries for stochastic numeric sketches."""

        if self._ag is None:
            raise RuntimeError("AG1 module is not loaded.")
        last_exc: Exception | None = None
        attempts = max(1, self.translation_retries)
        for attempt in range(1, attempts + 1):
            problem = self._ag.pr.Problem.from_txt(raw_problem, translate=True)
            try:
                graph, _ = self._ag.gh.Graph.build_problem(problem, self._ag.DEFINITIONS)
                return problem, graph, attempt
            except Exception as exc:  # pragma: no cover - rare numeric path.
                last_exc = exc
                if not self._is_transient_numeric_translation_error(type(exc).__name__):
                    raise
        raise RuntimeError(f"AG1 graph build failed after {attempts} attempts: {last_exc!r}")

    def _translate_aux_robust(self, raw_problem: str, lm_out: str, initial_graph: Any) -> tuple[str, dict[str, Any]]:
        """Translate AG1 LM output with robust numeric validation."""

        if self._ag is None:
            raise RuntimeError("AG1 module is not loaded.")
        attempts = max(1, self.translation_retries)
        first = self._ag.try_translate_constrained_to_construct(lm_out, initial_graph)
        if not first.startswith("ERROR:"):
            return first, {"translation_attempts": 1, "translation_retry_reason": ""}
        if not self._is_transient_numeric_translation_error(first):
            return first, {"translation_attempts": 1, "translation_retry_reason": "non_retryable"}

        last = first
        for attempt in range(2, attempts + 1):
            try:
                _, graph, _ = self._build_problem_graph_robust(raw_problem)
                translated = self._ag.try_translate_constrained_to_construct(lm_out, graph)
            except Exception as exc:  # pragma: no cover - defensive.
                translated = "ERROR: " + repr(exc)
            if not translated.startswith("ERROR:"):
                return translated, {
                    "translation_attempts": attempt,
                    "translation_retry_reason": "transient_numeric_sketch",
                }
            last = translated
            if not self._is_transient_numeric_translation_error(translated):
                return translated, {
                    "translation_attempts": attempt,
                    "translation_retry_reason": "became_non_retryable",
                }
        return last, {
            "translation_attempts": attempts,
            "translation_retry_reason": "transient_numeric_exhausted",
        }


def main() -> None:
    payload = json.loads(os.environ.get("AG1_LOCAL_PAYLOAD", '{"problem_id": "orthocenter"}'))
    service = AG1AuxService()
    print(json.dumps(service.handle(payload), indent=2))


if __name__ == "__main__":
    main()
