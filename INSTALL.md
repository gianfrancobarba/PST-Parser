# Installazione

Due percorsi. Il primo non richiede GPU e serve a ricalcolare le metriche o a lavorare sul codice;
il secondo serve ad addestrare e a generare predizioni.

I requisiti di sistema sono in [REQUIREMENTS.md](REQUIREMENTS.md).

## Percorso 1 — Solo valutazione

### Con Docker

```bash
docker build -f docker/Dockerfile --target eval -t pstparser:eval .
```

```bash
docker compose -f docker/compose.yaml --profile eval run --rm score score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

L'immagine non contiene PyTorch né alcuna libreria di modelli: pesa 913 MB e gira su qualsiasi
macchina.

### Senza Docker

```bash
uv sync --no-dev
```

```bash
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

## Percorso 2 — Training e generazione

### Con Docker

Richiede l'NVIDIA Container Toolkit installato sull'host. Verificarlo con:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Poi:

```bash
docker build -f docker/Dockerfile --target train -t pstparser:train .
```

```bash
docker compose -f docker/compose.yaml --profile train run --rm train prepare-data --config configs/experiments/baseline.yaml
```

```bash
docker compose -f docker/compose.yaml --profile train run --rm train train --config configs/experiments/baseline.yaml
```

I pesi del modello base vivono in un volume nominato, quindi vengono scaricati una sola volta e
sopravvivono alla ricostruzione dell'immagine. Le directory `data/`, `outputs/` e `results/` sono
montate dall'host, per cui gli artefatti restano disponibili a fine esecuzione.

### Senza Docker

```bash
uv sync --extra cu128 --extra train --extra unsloth
```

Su una macchina senza GPU, per sviluppare o eseguire i test, sostituire il primo extra:

```bash
uv sync --extra cpu --extra train --extra dev
```

I due extra `cpu` e `cu128` sono dichiarati incompatibili fra loro: `uv` rifiuta di installarli
insieme, così non è possibile ritrovarsi con una build di PyTorch diversa da quella voluta.

## Verifica dell'installazione

```bash
uv run pstparser validate-config --config configs/experiments/baseline.yaml
```

```bash
uv run pstparser prepare-data --config configs/experiments/baseline.yaml
```

La preparazione deve riportare **975 record**, 17 correzioni all'annotazione applicate, una
partizione di **783 / 87 / 87** dopo aver scartato 18 record duplicati, e **zero segnalazioni di
integrità**. Il controllo è bloccante: se un segmento non si ritrova nel prompt da cui è stato
estratto, o se un prompt resta scoperto oltre la soglia, la preparazione si ferma invece di
consegnare quei record al training.

Per la suite di test:

```bash
uv run pytest
```

```bash
uv run pytest -m slow
```

I test rapidi girano su CPU in meno di mezzo minuto. Quelli marcati `slow` eseguono la pipeline
completa con un modello giocattolo scaricato da Hugging Face, e richiedono connessione al primo
avvio.

## Problemi noti

**Il backend ottimizzato compila i propri kernel all'importazione, non all'installazione.**
L'installazione può quindi riuscire e il training fallire più tardi con un errore di CUDA. In quel
caso, `--set model.backend=hf` seleziona il percorso basato su transformers e peft, che è più lento
ma non applica alcuna modifica a runtime alle librerie sottostanti.

**`make` non è disponibile su Windows** salvo installarlo separatamente. Ogni target del `Makefile`
corrisponde a un comando `uv run` documentato nel [README](README.md).
