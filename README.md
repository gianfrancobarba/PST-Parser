# PST-Parser

Pipeline di fine-tuning supervisionato e valutazione per la **segmentazione strutturale di prompt**
secondo il framework Prompt Syntax Tree (PST).

Dato un prompt in linguaggio naturale, il modello ne estrae i segmenti testuali e li classifica nei
nodi di una tassonomia gerarchica, producendo un oggetto JSON:

```
prompt
├── main_instruction
├── context
│   ├── format          istruzioni positive sulla presentazione dell'output
│   ├── constrains      vincoli restrittivi sulla generazione
│   ├── data            contenuto grezzo da processare
│   └── role            definizione della persona
├── examples            coppie input-output per few-shot learning
└── reasoning
    ├── influence           trigger di ragionamento
    ├── reasoning_examples  esempi con passi intermedi espliciti
    └── paths               rami alternativi di ragionamento
```

## Requisiti

- Python 3.11 o 3.12
- [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente
- Per il training: GPU NVIDIA con almeno 16 GB di VRAM e driver compatibili con CUDA 12.8
- Un token Hugging Face con permesso di lettura, se il modello base è ad accesso condizionato

La sola valutazione non richiede GPU.

## Installazione

```bash
uv sync --extra cu128 --extra train --extra unsloth
```

Per una macchina senza GPU, sufficiente per preparare i dati e calcolare le metriche:

```bash
uv sync --extra cpu --extra dev
```

Copiare `.env.example` in `.env` e compilare i valori necessari.

## Uso

```bash
# Verifica che una configurazione sia ben formata, senza eseguire nulla
uv run pstparser validate-config --config configs/experiments/baseline.yaml
```

```bash
# Converte il corpus annotato in record e ne congela la partizione
uv run pstparser prepare-data --config configs/experiments/baseline.yaml
```

```bash
# Addestra gli adapter sul corpus preparato
uv run pstparser train --config configs/experiments/baseline.yaml
```

Ogni comando accetta `--set chiave.annidata=valore` per sovrascrivere puntualmente la
configurazione, ad esempio `--set training.max_steps=50`. Le sovrascritture sono registrate nella
configurazione risolta che accompagna ogni run.

Per una prova rapida senza GPU, sostituendo il modello con uno giocattolo:

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set model.backend=hf --set model.name=hf-internal-testing/tiny-random-LlamaForCausalLM --set model.load_in_4bit=false --set model.max_seq_length=512 --set training.max_steps=2 --set training.warmup_steps=0 --set training.optim=adamw_torch --set inference.max_new_tokens=32
```

## Artefatti di una run

Ogni esecuzione di `train` scrive una directory sotto `outputs/`:

| File | Contenuto |
|---|---|
| `adapter/` | Pesi LoRA e tokenizer |
| `run_manifest.json` | Commit git, configurazione risolta e suo digest, digest di corpus, record, system prompt, chat template e adapter, seed applicati, versioni delle librerie, descrizione dell'acceleratore |
| `training_metrics.jsonl` | Metriche emesse dal trainer, una per riga |

Il manifest è ciò che rende un risultato tracciabile: consente di risalire con esattezza a quale
codice, quali dati e quale ambiente lo hanno prodotto.

## Configurazione

Le configurazioni sono file YAML composti su due livelli:

- `configs/_base/` contiene i frammenti riutilizzabili (corpus, modello, training, valutazione);
- `configs/experiments/` contiene gli esperimenti, che dichiarano `extends` sui frammenti e
  sovrascrivono solo ciò che cambia.

`configs/experiments/baseline.yaml` è l'esperimento di riferimento: ogni altra configurazione
dovrebbe esprimersi come delta rispetto a questo, così che i confronti restino significativi.

La configurazione risolta viene serializzata insieme agli artefatti di ogni run, per cui è sempre
possibile risalire ai parametri esatti con cui un risultato è stato prodotto.

## Struttura

```
configs/      configurazioni degli esperimenti
data/raw/     corpus annotato, immutabile
data/splits/  identificativi delle partizioni, congelati
prompts/      system prompt versionato
src/pstparser/
  config/     schema tipizzato e composizione YAML
  pst/        tassonomia e costruzione dei target
  data/       lettura, normalizzazione, controllo di integrità, partizionamento
  models/     caricamento del modello base
  training/   ricetta di fine-tuning
  inference/  generazione delle predizioni
  evaluation/ metriche e report
  repro/      seeding e manifest di run
results/      predizioni e metriche di riferimento
tests/        suite di test
```

## Limiti noti

- **Il corpus disponibile non è quello usato nel lavoro originale.** Il notebook di partenza legge
  un file da circa 1208 righe che non è stato consegnato; questo repository contiene la versione da
  975 righe. I risultati numerici non sono quindi direttamente confrontabili con quelli pubblicati.
  Se il file originale venisse recuperato, è sufficiente aggiornare `source_path` e `sheet_name` in
  `configs/_base/data/`.
- **Tre nodi foglia su nove non sono supervisionati.** Il ramo `reasoning` conta tre esempi
  annotati in tutto: `reasoning_examples` e `paths` non hanno alcun esempio positivo, e le metriche
  per questi nodi non sono informative.

## Note sul porting

Alcuni elementi presenti nel notebook di partenza non sono stati portati:

- un secondo ciclo di inferenza, duplicato del primo ma privo di template conversazionale e di
  system prompt, e con un budget di generazione insufficiente a completare gli output;
- una cella priva di sorgente;
- un insieme di classi di validazione mai istanziate, che dichiaravano campi incoerenti con i
  target effettivamente prodotti. Lo schema di validazione viene generato dalla tassonomia, quindi
  è coerente per costruzione;
- dipendenze installate e mai utilizzate;
- comandi shell e autenticazione interattiva, sostituiti da equivalenti programmatici e da
  variabili d'ambiente.

## Licenza

MIT. Vedi [LICENSE](LICENSE).
