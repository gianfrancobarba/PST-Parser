# Requirements and operating limits

## Software

The package declares Python `>=3.11,<3.13`; `.python-version` selects 3.11. Use `uv` and `uv.lock`,
and check the dependencies resolved for the system, the interpreter and the extras. The lock does not
make kernels, drivers and hardware identical. Docker and the NVIDIA Container Toolkit are needed for
the containerised path with a GPU; check the driver and the CUDA build with `nvidia-smi` and a smoke
test of the backend. Git, when available, provides read-only metadata: no staging, commit or push is
needed.

## Hardware and memory

Preparation, PST scoring and the paired analysis load no model and can run on a CPU. Their duration
depends on the records. Code generated downstream must run in an isolated environment configured
separately: the training image is not that sandbox by itself.

Training and generation depend on model, quantisation, batch, sequences and backend. An 8 GB GPU
such as the RTX 4070 laptop used in development **does not guarantee that the baseline fits**. Check
peak memory and duration on the long cases; 32 GB of RAM do not replace VRAM without costs or limits.
Run the parser and the generator separately.

To reduce the batch while keeping the nominal number of examples per update:

```text
--set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8
```

This **does not guarantee an equivalent training trajectory**: padding, tokens per batch, numerical
reductions and implementation may change. Record the actual configuration.

The earlier length measurements on 1,123 records describe a snapshot. After changes to the corpus,
the tokenizer, the system prompt or the chat template, recompute inputs, targets and sequences.
Reducing `model.max_seq_length` can truncate data: measure which records and how many. A wide window
can affect memory depending on the backend. `inference.max_new_tokens` is a limit, not a guarantee of
complete JSON. Measure disk space and downloads for the actual weights, checkpoints, images and caches,
without taking earlier values in GB as fixed.

## Credentials

`HF_TOKEN` is needed when the model repository asks for it, possibly after accepting its access
conditions. `NVIDIA_API_KEY` is for the configured synthetic-prompt provider. Copy `.env.example` to
`.env` and fill in only what you need. Never put credentials in shared artefacts.

## Reproducibility and scientific status

Model-free transformations repeat with the same inputs, configuration and a compatible environment.
Seeds and greedy decoding **do not guarantee bit-for-bit determinism** across versions, hardware or
backends. The manifest records seeds and settings; it is not a complete detector of every
non-deterministic operation. Hashes and per-stage manifests improve traceability, without
guaranteeing that every external artefact can be recovered.

The empirical study was run from the frozen protocol (`protocol/`, commit `394c893`); its deviations
are dated in `protocol/deviations.md`. The determinism of the generation was measured on a 5% sample
of the greedy requests regenerated in a separate session (`protocol/determinism/`). Historical outputs
at the root of `results/` do not prove that the current configuration was run. Follow
[INSTALL.md](INSTALL.md) for fresh directories and separate splits.

Colab gives no guarantee of a specific GPU or of unlimited hours; the runs of the study record the GPU
they ran on, and a final run refuses any other.
