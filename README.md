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
│   ├── constraints     vincoli restrittivi sulla generazione
│   ├── data            contenuto grezzo da processare
│   └── role            definizione della persona
├── examples            coppie input-output per few-shot learning
└── reasoning
    ├── influence           trigger di ragionamento
    ├── reasoning_examples  esempi con passi intermedi espliciti
    └── paths               rami alternativi di ragionamento
```

Ogni foglia è una **lista di segmenti**: una categoria può comparire in più punti del prompt, e
tenerne i pezzi separati è ciò che permette di ricostruire il testo di partenza. Come la
discontinuità venga segnata dipende dal formato in cui il corpus è annotato: in un foglio di calcolo
con il token `<sep>`, perché una cella regge una stringa sola; in un file di testo scrivendo un
segmento per voce di lista, perché una sequenza non ha bisogno di un token che dica dove si divide.

## Requisiti

- Python 3.11 o 3.12
- [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente
- Per il training: GPU NVIDIA con almeno 8 GB di VRAM e driver compatibili con CUDA 12.8
- Un token Hugging Face con permesso di lettura, se il modello base è ad accesso condizionato

La sola valutazione non richiede GPU.

## Installazione

```bash
uv sync --extra cu128 --extra train --extra unsloth
```

Per una macchina senza GPU, sufficiente per preparare i dati e calcolare le metriche:

```bash
uv sync --extra cpu --extra train --extra dev
```

Copiare `.env.example` in `.env` e compilare i valori necessari.

Istruzioni complete, incluso il percorso containerizzato, in [INSTALL.md](INSTALL.md); requisiti
di sistema in [REQUIREMENTS.md](REQUIREMENTS.md).

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

```bash
# Genera le predizioni per lo split di valutazione
uv run pstparser generate --config configs/experiments/baseline.yaml --adapter outputs/<run>/adapter
```

```bash
# Calcola le metriche. Non richiede GPU né modello.
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

```bash
# Colloca ogni frase generata nel prompt e ne inietta la posizione. Nemmeno questo richiede GPU.
uv run pstparser align --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

Ogni comando accetta `--set chiave.annidata=valore` per sovrascrivere puntualmente la
configurazione, ad esempio `--set training.max_steps=50`. Le sovrascritture sono registrate nella
configurazione risolta che accompagna ogni run.

I valori sono interpretati come YAML, dove `no`, `yes`, `on` e `off` sono booleani. Per passare una
di quelle parole come testo va virgolettata: `--set 'training.eval_strategy="no"'`. Un tipo
sbagliato viene comunque rifiutato dalla validazione prima che il run inizi.

Su una GPU con 8 GB di VRAM, dimezzare il batch e raddoppiare l'accumulo lascia invariata la
dimensione effettiva, e quindi la traiettoria di ottimizzazione:

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8
```

Per una prova rapida senza GPU, sostituendo il modello con uno giocattolo:

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set model.backend=hf --set model.name=hf-internal-testing/tiny-random-LlamaForCausalLM --set model.load_in_4bit=false --set model.max_seq_length=512 --set training.max_steps=2 --set training.warmup_steps=0 --set training.optim=adamw_torch --set inference.max_new_tokens=32
```

## Esecuzione in container

Due immagini, costruite dallo stesso `Dockerfile` con target diversi.

| Target | Base | Contenuto | Dimensione |
|---|---|---|---|
| `train` | Python 3.11 slim | Stack completo di fine-tuning, richiede GPU | 12.8 GB |
| `eval` | Python 3.11 slim | Solo ciò che serve al calcolo delle metriche | 913 MB |

```bash
docker build -f docker/Dockerfile --target eval -t pstparser:eval .
```

```bash
docker compose -f docker/compose.yaml --profile eval run --rm score score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

L'immagine di valutazione **non contiene PyTorch né alcuna libreria di modelli**. Non è una
questione di dimensioni: è ciò che rende le metriche ricalcolabili anche quando l'ambiente di
training non è riproducibile, e permette di verificare i risultati su una macchina qualsiasi.

L'immagine è autosufficiente: configurazioni e system prompt sono al suo interno, e l'unica cosa
da fornire è il file di predizioni.

```bash
docker run --rm -v "$PWD/results:/work" pstparser:eval score --config configs/experiments/baseline.yaml --predictions /work/predictions.jsonl
```

Il training richiede l'NVIDIA Container Toolkit:

```bash
docker compose -f docker/compose.yaml --profile train run --rm train train --config configs/experiments/baseline.yaml
```

I pesi del modello base risiedono in un volume nominato, quindi non entrano mai nell'immagine e
sopravvivono alle ricostruzioni. `data/`, `outputs/` e `results/` sono montate dall'host.


## Artefatti di una run

Ogni esecuzione di `train` scrive una directory sotto `outputs/`:

| File | Contenuto |
|---|---|
| `adapter/` | Pesi LoRA e tokenizer |
| `run_manifest.json` | Commit git, configurazione risolta e suo digest, digest di corpus, record, system prompt, chat template e adapter, seed applicati, versioni delle librerie, descrizione dell'acceleratore |
| `training_metrics.jsonl` | Metriche emesse dal trainer, una per riga |

Il manifest è ciò che rende un risultato tracciabile: consente di risalire con esattezza a quale
codice, quali dati e quale ambiente lo hanno prodotto.

`generate`, `score` e `align` producono a loro volta `predictions.jsonl`, `results.json`,
`eval_details.jsonl` e `alignments.jsonl`.

## Valutazione

La generazione è separata dal calcolo delle metriche. `generate` è l'unico stadio che richiede un
acceleratore e persiste le predizioni; `score` legge quel file e non carica alcun modello, quindi
gira ovunque in pochi secondi. Le predizioni di riferimento sono versionate in `results/`, per cui
ogni numero è ricalcolabile senza GPU.

Le metriche di contenuto sono calcolate sulle predizioni che risultano ben formate:

| Metrica | Significato | Valore atteso |
|---|---|---|
| Validità JSON | quota di predizioni che si parsificano **e** rispettano lo schema | alto |
| F1 per campo | accordo token per token con il riferimento, per nodo foglia, con il support | alto |
| Matrice di confusione | a quale nodo è stato assegnato ogni token del prompt, contro quello corretto | diagonale |
| Distanza di edit fra alberi | quante modifiche separano la struttura predetta da quella attesa | basso |
| Tasso di allucinazione | quota di token predetti assenti dal prompt | zero |
| Copertura | quota di token del prompt collocati in un nodo | uno |
| Ricostruzione esatta | quota di prompt che si riottengono concatenando le foglie in ordine | uno |
| Frasi collocate | quota di frasi generate ritrovate nel prompt | uno |

### Ricostruzione e posizioni

Il modello genera solo il testo delle foglie. Un secondo stadio, deterministico e senza modello,
ricolloca ogni frase nel prompt da cui è stata estratta e ne ricava la posizione: ogni frase
rivendica un intervallo di caratteri, due intervalli non possono sovrapporsi, e la posizione è il
rango dell'intervallo. Da lì discende l'artefatto con foglie `{"position": ..., "phrase": ...}` e la
verifica diretta che il prompt si riottenga concatenando le foglie in ordine.

La copertura è solo un sostituto di quella proprietà: dice quali parole sono state collocate, mai in
che ordine. Un output che rimescolasse completamente le frasi otterrebbe comunque copertura piena.

Il comportamento di ciascuna metrica è bloccato da casi golden in `tests/metrics/cases/`, con i
punteggi attesi versionati accanto: una variazione non spiegata è un difetto, non un cambiamento
silenzioso di significato.

### Decoding vincolato

Impostando `inference.structured_output: true` (richiede l'extra `structured`) la generazione è
vincolata allo schema derivato dalla tassonomia, e la validità JSON diventa garantita per
costruzione anziché misurata.

Va usato con una avvertenza metodologica: in quella modalità la validità JSON smette di dire
alcunché sul fine-tuning, perché è imposta dall'esterno. Il valore sta nel confronto fra le due
esecuzioni — libera per misurare quanto il modello ha interiorizzato il formato, vincolata per
valutare la qualità del contenuto a parità di formato.

## Espansione del corpus

Il corpus proviene da conversazioni reali fra sviluppatori e assistente, dove le richieste dirette
dominano: i rami della tassonomia che descrivono il ragionamento sono quasi privi di esempi. Il
comando `synth` genera prompt aggiuntivi per quei paradigmi, partendo da trenta seed scritti a mano
in `data/seeds/paradigms.yaml`.

```bash
uv run pstparser synth --config configs/experiments/baseline.yaml
```

Il provider è un qualunque servizio con endpoint di chat completions compatibile con il formato
OpenAI; la credenziale è letta dall'ambiente, mai da un file di configurazione.

**Il comando produce prompt, non annotazioni.** L'output è un file con il prompt e il paradigma
compilati e le nove foglie vuote, nello stesso formato che `prepare-data` legge: va etichettato a
mano prima di poter rientrare. Questo passaggio resta lavoro umano.

Il formato coincide di proposito. Trascrivere 150 prompt da un formato a un altro sono 150 occasioni
di rompere l'estrazione esatta, e il controllo a valle riporterebbe un errore di trascrizione come
se fosse un errore di annotazione.

Il file annotato non va fuso nel corpus di partenza: si aggiunge a `data.sources`, che li concatena
mantenendoli file distinti. Il materiale consegnato resta separato da quello annotato dopo, e
`prepare-data` riporta quanti record ha portato ciascuna sorgente.

## Notebook Colab

`notebooks/colab.ipynb` clona il repository e invoca la stessa interfaccia a riga di comando usata
in locale e in container. Non contiene logica: logica duplicata sarebbe la prima cosa a divergere
dal resto del progetto.

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
configs/         configurazioni degli esperimenti
docker/          Dockerfile multi-target e compose
data/raw/        corpus consegnato, immutabile
data/annotated/  annotazioni scritte qui, come testo modificabile
data/splits/     identificativi delle tre partizioni, congelati
prompts/         system prompt versionato
src/pstparser/
  config/        schema tipizzato e composizione YAML
  pst/           tassonomia, serializzazione e schema derivato
  data/          lettura, normalizzazione, integrità, partizioni, formati di scambio
  models/        caricamento del modello base e degli adapter
  training/      ricetta di fine-tuning
  inference/     generazione delle predizioni
  evaluation/    metriche e report
  repro/         seeding e manifest di run
results/         predizioni e metriche di riferimento
tests/           suite di test
```

Le dipendenze fra i package sono aciclee e orientate verso il basso: `pst/` non dipende da nulla,
`data/` dipende da `pst/`, `evaluation/` dipende da `data/`, e solo `models/`, `training/` e
`inference/` toccano PyTorch. È questa direzione a rendere possibile un ambiente di valutazione
privo dello stack di modelli.

## Partizione del corpus

Tre insiemi, non due. Sorvegliare un insieme durante il training e poi riportare i risultati sullo
stesso insieme non misura la generalizzazione: misura quanto bene si è fermato il training. Il
**validation** è ciò che il ciclo di addestramento può guardare; il **test** è ciò da cui si citano
i risultati, e si legge una volta sola.

I record che condividono un prompt finiscono dalla stessa parte. Il corpus contiene prompt ripetuti,
e lasciarne uno in training e un altro in test permetterebbe di essere valutati su testo già visto.
I record identici sia nel prompt sia nell'annotazione vengono ridotti a uno.

`generate` produce predizioni per il **test** in modo predefinito; `--split val` serve alle verifiche
fatte durante lo sviluppo.

## Formati del corpus

Una sorgente dichiara come è scritta. Dedurlo dal suffisso farebbe dipendere il comportamento da come
il file è stato chiamato, e direbbe la stessa cosa due volte — nell'estensione e nella presenza di un
foglio — senza nulla che tenga d'accordo le due.

| | `excel` | `yaml` |
|---|---|---|
| Cos'è | Il foglio di calcolo consegnato | Annotazioni scritte in questo repository |
| `sheet` | obbligatorio | assente |
| `fixes` | ammesse | vietate: si corregge il file |
| Foglia indirizzata da | intestazione di colonna | percorso nella tassonomia |
| Segmenti disgiunti | token `<sep>` nella cella | una voce di lista ciascuno |

```yaml
data:
  sources:
    - path: data/raw/prompt_dataset_v2.xlsx
      sheet: prompt_dataset_v2
      fixes: data/raw/annotation_fixes.yaml
    - path: data/annotated/reasoning.yaml
      format: yaml
```

Un record annotato come testo si legge per intero in un diff, che è la ragione per cui il formato
esiste: una revisione di cento annotazioni fatta riga per riga non è la stessa cosa di una fatta
aprendo un binario. E nominando le foglie per percorso, un file scritto qui non eredita
l'intestazione `CONSTRAINS` che il foglio consegnato porta per compatibilità.

```yaml
records:
  - paradigm: tree_of_thoughts
    prompt: |-
      class A: pass

      class B: pass

      Rename these classes to something meaningful.
      Do not change their behaviour.
    leaves:
      main_instruction: Rename these classes to something meaningful.
      context.constraints: Do not change their behaviour.
      context.data:
        - "class A: pass"
        - "class B: pass"
```

Una foglia omessa non ha segmenti; una stringa è un segmento; una lista è un segmento per voce. Il
prompt va scritto con uno scalare **letterale** (`|-`): quello ripiegato (`>-`) trasforma le andate a
capo in spazi, e il testo cambierebbe senza dirlo.

## Correzioni all'annotazione

Il foglio di calcolo in `data/raw/` è un input e resta esattamente come è stato consegnato. Le
correzioni vivono in `data/raw/annotation_fixes.yaml`, ciascuna con la riga a cui si applica e il
motivo per cui è stata fatta: virgolette aggiunte durante il copia-incolla, un carattere digitato per
un altro, un passaggio riscritto in prima persona, una cella che il foglio di calcolo ha letto come
formula e distrutto. Una correzione che non descrive più la sua cella è un errore, non un'operazione
a vuoto, quindi il corpus e le sue correzioni non possono divergere senza che qualcuno se ne accorga.

Il meccanismo vale per le sole sorgenti a foglio di calcolo, ed è una restrizione dichiarata dallo
schema. Esiste perché quel file non si può modificare; un'annotazione scritta qui si corregge dove
sta, e la storia di ciò che è cambiato è già il diff. Due posti in cui cercare la verità di un record
sono peggio di uno.

## Limiti noti

- **Il corpus disponibile non è quello usato nel lavoro originale.** Il notebook di partenza legge
  un file da circa 1208 righe che non è stato consegnato; questo repository contiene la versione da
  975 righe. I risultati numerici non sono quindi direttamente confrontabili con quelli pubblicati.
  Se il file originale venisse recuperato, basta aggiungerlo a `data.sources` in
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
