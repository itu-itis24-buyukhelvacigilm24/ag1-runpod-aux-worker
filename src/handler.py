from __future__ import annotations

import json
import traceback

import runpod

try:
    from ag1_service import AG1AuxService
except ModuleNotFoundError:  # Allows importing as src.handler from repo root.
    from .ag1_service import AG1AuxService


_SERVICE: AG1AuxService | None = None


def get_service() -> AG1AuxService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AG1AuxService()
    return _SERVICE


def handler(job):
    payload = job.get("input", job) or {}
    try:
        if payload.get("healthcheck"):
            return get_service().healthcheck(load_model=bool(payload.get("load_model", False)))
        return get_service().handle(payload)
    except Exception as exc:  # pragma: no cover - serverless boundary
        return {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=30),
            "payload_keys": sorted(payload.keys()),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
