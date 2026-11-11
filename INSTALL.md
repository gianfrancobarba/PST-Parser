# Installazione e verifiche

Requisiti in [REQUIREMENTS.md](REQUIREMENTS.md). Eseguire dalla radice del progetto.
`outputs/new-run/` è un esempio: scegliere un nome nuovo per ogni preparazione e usare gli
stessi override in train e generate. Non sovrascrivere split o risultati storici.
La preparazione tecnica downstream non equivale all'approvazione del protocollo da parte del docente.

## Preparazione e scoring CPU

```bash
uv sync --no-dev
uv run pstparser validate-config --config configs/experiments/baseline.yaml
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

Senza `--run-dir`, score crea una directory `score-...` sotto `training.output_dir` (`outputs/`
di default), mai accanto alle predizioni, con `results.json`, `eval_details.jsonl` e
`score_manifest.json`; align fa lo stesso con `align-...`. È un ricalcolo con metriche
attuali di predizioni storiche, non un nuovo training. Le collisioni con output esistenti sono rifiutate.

## Training e generazione

```bash
uv sync --extra cu128 --extra train --extra unsloth
```

Per sviluppo CPU con stack dei modelli e test:

```bash
uv sync --extra cpu --extra train --extra dev
```

Gli extra cpu e cu128 sono incompatibili. Il backend ottimizzato può compilare kernel
all'importazione: installazione riuscita non significa training validato.
`--set model.backend=hf` seleziona il percorso alternativo, senza garantire identità numerica
o gli stessi requisiti. Verificare modello e precisione effettivi con uno smoke.

```bash
uv run pstparser prepare-data --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --run-dir outputs/new-run/training
uv run pstparser generate --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --adapter outputs/new-run/training/adapter --run-dir outputs/new-run/generation
uv run pstparser score --config configs/experiments/baseline.yaml --predictions outputs/new-run/generation/predictions.jsonl --run-dir outputs/new-run/scoring
```

Sul corpus descritto dall'audit: 1123 record prima della deduplica (975 Excel e 148 YAML),
17 correzioni, 18 duplicati e split 905/100/100 sono dati dello snapshot, da verificare sul
report nuovo; non requisiti invarianti per corpus futuri. Il gate verifica collocazione/copertura
con soglie e normalizzazioni configurate: non certifica semantica o copertura strict al 100%.

prepare-data richiede directory fresche per record e split. I manifest nuovi sono distinti:
`prepare_manifest.json`, `train_manifest.json`, `generate_manifest.json`, `score_manifest.json`.
Gli override dei percorsi devono accompagnare train/generate per selezionare gli stessi dati.

## Audit del corpus e moduli downstream

Si invocano come moduli Python, senza entry point dedicati; non richiedono GPU tranne
`empirical generate`, che carica il modello.

```bash
uv run python -m pstparser.data.audit --config configs/experiments/baseline.yaml --output outputs/new-run/corpus-audit.json
uv run python -m pstparser.empirical --help
```

L'audit è in sola lettura e rifiuta un file di output esistente; la directory deve esistere.
I sottocomandi `render`, `generate`, `check` e `analyze` di `pstparser.empirical` richiedono input
espliciti (annotazioni canoniche, configurazione congelata, oracoli, disegno dell'analisi).
`check` richiede Docker e un'immagine Python già presente localmente, fissata per digest
(`nome@sha256:...`): non scarica immagini né esegue codice generato sull'host.

## Container

```bash
docker build -f docker/Dockerfile --target eval -t pstparser:eval .
docker compose -f docker/compose.yaml --profile eval run --rm score score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

L'immagine eval non richiede lo stack dei modelli. Dimensione e compatibilità dipendono dalla build.
Per GPU verificare NVIDIA Container Toolkit e runtime sull'host:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
docker build -f docker/Dockerfile --target train -t pstparser:train .
docker compose -f docker/compose.yaml --profile train run --rm train prepare-data --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-container-run/processed --set data.split.output_dir=outputs/new-container-run/splits
docker compose -f docker/compose.yaml --profile train run --rm train train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-container-run/processed --set data.split.output_dir=outputs/new-container-run/splits --run-dir outputs/new-container-run/training
```

Compose monta dati/output/risultati e una cache dei modelli. Queste immagini non costituiscono
automaticamente il sandbox downstream per codice non fidato.

## Test

```bash
uv run pytest -m "not slow and not gpu"
uv run pytest -m slow
```

I test slow possono scaricare un modello e richiedere rete/stack dei modelli. Durata e supporto
GPU dipendono dall'ambiente. Se la creazione dei temporanei è impedita, passare
`--basetemp outputs/test-temp-new` con una directory dedicata nuova: pytest la gestisce
come temporanea, quindi non scegliere percorsi con dati da conservare.

Su Windows make richiede installazione separata; i comandi CLI sopra non ne dipendono.
