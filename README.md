# AlphaGeometry 1 RunPod Aux Worker

[![Runpod](https://api.runpod.io/badge/itu-itis24-buyukhelvacigilm24/ag1-runpod-aux-worker)](https://console.runpod.io/hub/itu-itis24-buyukhelvacigilm24/ag1-runpod-aux-worker)

RunPod Serverless worker for the public AlphaGeometry 1 language model auxiliary
construction step.

This repository is intentionally standalone and public-safe: it does not contain
API keys, private geometry datasets, product code, or model weights. The Docker
image clones the public AlphaGeometry and Meliad repositories, then downloads the
public checkpoint from Hugging Face during image build or worker startup.

## What This Worker Does

Input:

```json
{
  "input": {
    "problem_id": "orthocenter",
    "beam_size": 4,
    "batch_size": 4,
    "sequence_length": 96
  }
}
```

or:

```json
{
  "input": {
    "problem": "my_problem\na b c = triangle; h = on_tline b a c, on_tline c a b ? perp a h b c"
  }
}
```

Output:

```json
{
  "ok": true,
  "elapsed_sec": 12.3,
  "public_parameter_count": "152M",
  "effective_batch_size": 4,
  "candidate_capacity": 4,
  "beam_truncated": false,
  "count": 1,
  "results": [
    {
      "problem_id": "orthocenter",
      "suggestions": [
        {
          "rank": 1,
          "lm_output": "e : C a c e 02 C b d e 03 ;",
          "translation": "e = on_line e a c, on_line e b d",
          "valid": true,
          "score": -1.23
        }
      ]
    }
  ]
}
```

Important runtime details:

- This worker runs the public AG1 auxiliary-construction LM checkpoint, not the
  full AG2 Gemini language-model/tree-search system from the AG2 paper.
- The public AG1 generation config is the 152M parameter model.
- AG1/Meliad returns one candidate per configured batch slot. Keep
  `batch_size >= beam_size`; the worker now enforces this internally with
  `effective_batch_size = max(batch_size, beam_size)`.
- `rank` preserves the raw AG1 beam output order. The numeric `score` is kept
  for diagnostics but is not used to reorder suggestions after grammar
  translation.

This does not solve metric length questions by itself. It proposes synthetic
auxiliary constructions. A full geometry solver stack should still be:

```text
formal problem
-> AG2 DDAR / synthetic closure
-> if stuck, AG1 aux worker
-> AG2 DDAR again with aux facts
-> metric value sidecar for lengths/angles/areas
-> teaching-step compiler
```

## RunPod Deploy

Use RunPod's GitHub integration so your local machine does not need Docker:

1. Create a public GitHub repository from this folder.
2. In RunPod Console, connect GitHub under Settings -> Connections.
3. Create a Serverless endpoint from GitHub.
4. If RunPod's build context is the repository root, set Dockerfile path to:

```text
Dockerfile
```

5. Recommended first GPU: RTX 4090 / L40S class.
6. Set endpoint timeout high for first test, for example 600 seconds.
7. Set max workers 1 for first debug run.
8. Optional env vars:

```text
AG1_HF_REPO=abrahamabelboodala/ALPHAGEOMETRY_ag_ckpt_vocab
AG1_CKPT_DIR=/runpod-volume/ag1_ckpt_vocab_hf
AG1_BEAM_SIZE=4
AG1_BATCH_SIZE=4
AG1_SEQUENCE_LENGTH=96
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

If no network volume is attached, the Docker image predownloads the checkpoint
into `/opt/ag_ckpt_vocab_hf`. For production, prefer a RunPod network volume so
new image builds do not carry model weights in every layer.

## Why Not RunPod Flash First

RunPod Flash is the nicer no-Docker route for pure Python apps, but this worker
has an old JAX/Flax/Meliad stack plus a roughly 1.2 GB checkpoint. A remote
Dockerfile build is the more controlled route for this specific AG1 model.

## Local Smoke Check

You can check only Python syntax locally:

```powershell
python -m py_compile .\handler.py .\src\ag1_service.py .\src\handler.py
```

Full model execution should run on Linux GPU. CPU execution works for syntax and
import checks but is too slow for normal use.

## Security

Do not commit `.env` files, RunPod API keys, OpenAI keys, Hugging Face tokens, or
private problem datasets. Configure secrets only in RunPod endpoint settings.

## License

This worker code is Apache-2.0. It builds on public Apache-2.0 projects:

- AlphaGeometry: https://github.com/google-deepmind/alphageometry
- Meliad: https://github.com/google-research/meliad
