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

## Stato del progetto e della valutazione

La pipeline PST e i moduli downstream sono distinti: `src/pstparser/empirical/` contiene
contratti, rendering, generazione, oracoli e analisi per il confronto fra le condizioni, ma i test
sintetici non sono risultati scientifici di generazione del codice. Il protocollo di valutazione
empirica è tenuto nella documentazione locale di progettazione, non versionata.

**Decisioni del relatore.** Le quattro decisioni su cui poggia il disegno:

- D1: prima il PST verificato a mano (H-gold), anche su un sottoinsieme di prompt; poi il PST
  prodotto dal parser (H-auto);
- D2: BigCodeBench-Instruct per la correttezza;
- D3: CodeQL per la sicurezza, con la sua inaffidabilità dichiarata come minaccia alla validità;
- D4: il secondo revisore, per le annotazioni e per i giudizi di sicurezza, è opzionale.

**A che punto è la valutazione.** I campioni e le esclusioni sono congelati in `protocol/`: le
esclusioni di BigCodeBench, i task idonei e quelli Hard sono in `protocol/oracles/bcb-exclusions.json`
(`excluded_count`, `eligible`, `eligible_hard`), il campione finale in `protocol/samples/bcb-final.json`
(`n`) e il pool pilota in `protocol/samples/pilot-bcb.json` e `pilot-llmseceval.json`. Il freeze del
protocollo (`protocol/protocol-freeze.json`) non è ancora stato eseguito: finché manca, ogni output
downstream è una prova tecnica, non un risultato.

**La run del parser `parser-2026-11-12-r3`** è quella che H-auto userà. I suoi artefatti
(`run-identity.json`, `run_manifest.json`, `results.json`) sono su Google Drive e non sono
versionati: identità e numeri della run vanno citati da quei file, una volta archiviati, e non sono
riprodotti qui. Due avvertenze di provenienza valgono già:

- **Unsloth.** Il backend ottimizzato carica un proprio repository pre-quantizzato al posto della
  revisione richiesta, quindi la revisione del modello di quella run non è verificata
  (`revision_verified: false`). La generazione downstream usa il backend `hf`, che verifica la
  revisione e rifiuta le run finali senza verifica.
- **CRLF.** L'archivio sorgente di quella run è stato creato da un checkout Windows con
  `core.autocrlf=true`: gli hash di alcuni file di testo nel manifest sono quelli delle copie CRLF,
  non dei blob git. Il modello ha comunque ricevuto testo LF, perché i file sono letti con
  `read_text`. Gli archivi nuovi si creano con `make source-archive`, che usa
  `git -c core.autocrlf=false archive` e verifica l'albero.

Gli artefatti in `results/` sono storici: non attestano che le configurazioni attuali siano state
addestrate e valutate. I ricalcoli vanno scritti in nuove directory, senza sostituire quelle prove.
Le run downstream vanno in `results/downstream/<run_id>/`.

## Requisiti

- Python 3.11 o 3.12
- [`uv`](https://docs.astral.sh/uv/) per la gestione dell'ambiente
- Per il training: GPU NVIDIA e stack CUDA compatibile; 8 GB richiedono uno smoke di memoria e tempi e non garantiscono la fattibilità della configurazione base
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

Gli esempi usano `outputs/new-run/`: scegliere un nome nuovo. La preparazione rifiuta record
o split già presenti; train e generate devono ricevere gli stessi override dei percorsi.
Gli esempi di training sono alternativi, non da eseguire tutti nella medesima directory.

```bash
# Verifica che una configurazione sia ben formata, senza eseguire nulla
uv run pstparser validate-config --config configs/experiments/baseline.yaml
```

```bash
# Converte il corpus annotato in record e ne congela la partizione
uv run pstparser prepare-data --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits
```

```bash
# Addestra gli adapter sul corpus preparato
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --run-dir outputs/new-run/training
```

```bash
# Genera le predizioni per lo split di valutazione
uv run pstparser generate --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --adapter outputs/new-run/training/adapter --run-dir outputs/new-run/generation
```

```bash
# Senza adapter risponde il modello base, non addestrato, sullo stesso test set
uv run pstparser generate --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --run-dir outputs/new-run/zero_shot
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

Su una GPU con 8 GB di VRAM, dimezzare il batch e raddoppiare l'accumulo conserva il numero
nominale di esempi per aggiornamento, ma **non garantisce la stessa traiettoria di ottimizzazione**:
padding, riduzioni numeriche, ordine dei batch e implementazione possono incidere.

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8 --run-dir outputs/new-run/training
```

Per una prova rapida senza GPU, sostituendo il modello con uno giocattolo:

```bash
uv run pstparser train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits --set model.backend=hf --set model.name=hf-internal-testing/tiny-random-LlamaForCausalLM --set model.load_in_4bit=false --set model.max_seq_length=512 --set training.max_steps=2 --set training.warmup_steps=0 --set training.optim=adamw_torch --set inference.max_new_tokens=32 --run-dir outputs/new-run/training
```

## Esecuzione in container

Due immagini, costruite dallo stesso `Dockerfile` con target diversi.

| Target | Base | Contenuto |
|---|---|---|
| `train` | Python 3.11 slim | Stack completo di fine-tuning, richiede GPU |
| `eval` | Python 3.11 slim | Solo ciò che serve al calcolo delle metriche |

Le dimensioni delle immagini dipendono dalla build e dalle dipendenze effettive.

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
docker compose -f docker/compose.yaml --profile train run --rm train train --config configs/experiments/baseline.yaml --set data.processed_dir=outputs/new-run/processed --set data.split.output_dir=outputs/new-run/splits
```

I pesi del modello base risiedono in un volume nominato, quindi non entrano mai nell'immagine e
sopravvivono alle ricostruzioni. `data/`, `outputs/` e `results/` sono montate dall'host.


## Artefatti di una run

Ogni esecuzione di `train` scrive una directory sotto `outputs/`:

| File | Contenuto |
|---|---|
| `adapter/` | Pesi LoRA e tokenizer |
| `train_manifest.json` | Manifest v2 dello stadio: configurazione, hash degli input registrati e snapshot del codice, seed, versioni e metadati disponibili del training |
| `training_metrics.jsonl` | Metriche emesse dal trainer, una per riga |

I nuovi manifest sono distinti per stadio: `prepare_manifest.json`, `train_manifest.json`,
`generate_manifest.json`, `score_manifest.json` e `align_manifest.json`. La preparazione salva
il proprio manifest nella directory dei dati preparati. `run_manifest.json` rimane un nome
presente negli artefatti storici, non quello delle nuove run. Gli hash e la configurazione
migliorano la tracciabilità, ma non garantiscono riproducibilità bit per bit né provano da soli
che tutti gli artefatti esterni siano recuperabili. Un digest nullo indica un input non disponibile.

`generate`, `score` e `align` producono a loro volta `predictions.jsonl`, `results.json`,
`eval_details.jsonl` e `alignments.jsonl`. Senza `--run-dir`, `score` e `align` creano una nuova
sottodirectory `score-...` o `align-...` sotto `training.output_dir` (`outputs/` di default), mai
accanto alle predizioni: così un ricalcolo non scrive dentro `results/`. Le collisioni con
artefatti dello stesso stadio sono rifiutate: scegliere una directory nuova, non cancellare o sovrascrivere le run storiche.

## Valutazione

La generazione è separata dal calcolo delle metriche. `generate` carica il modello e persiste
le predizioni; `score` legge quel file senza caricare modelli. Tempo e memoria dipendono dai dati.
Le predizioni storiche in `results/` consentono ricalcoli CPU, distinguendo la versione delle metriche.

Le metriche di contenuto sono calcolate sulle predizioni che risultano ben formate:

| Metrica | Significato | Valore atteso |
|---|---|---|
| Validità JSON | quota di predizioni che si parsificano **e** rispettano lo schema | alto |
| F1 per campo | accordo token per token con il riferimento, per nodo foglia, con il support | alto |
| Matrice di confusione | a quale nodo è stato assegnato ogni token del prompt, contro quello corretto | diagonale |
| Distanza di edit fra alberi | quante modifiche separano la struttura predetta da quella attesa | basso |
| Tasso di allucinazione | quota di token predetti assenti dal prompt | zero |
| Copertura | quota di token del prompt collocati in un nodo | uno |
| Ricostruzione legacy normalizzata | ricostruzione del sottoinsieme allineato con normalizzazione; non prova fedeltà completa | alto |
| Tutte le foglie contabilizzate (v2) | quota di predizioni senza foglie estratte rimaste non collocate | uno |
| Ricostruzione completa strict (v2) | tutte le foglie contabilizzate e ricostruzione testuale esatta | uno |
| Frasi collocate | quota di frasi generate ritrovate nel prompt | uno |

Nessuno di questi numeri dice però quanto il fine-tuning abbia aggiunto: per quello serve un termine
di paragone. `generate` senza `--adapter` fa rispondere il modello base allo stesso test set e con
lo stesso system prompt, ed è il confronto più economico disponibile — non richiede training.

### Ricostruzione e posizioni

Il modello genera solo il testo delle foglie. Un secondo stadio, deterministico e senza modello,
ricolloca ogni frase nel prompt da cui è stata estratta e ne ricava la posizione: ogni frase
rivendica un intervallo di caratteri, due intervalli non possono sovrapporsi, e la posizione è il
rango dell'intervallo. Da lì discende l'artefatto con foglie `{"position": ..., "phrase": ...}` e la
verifica di ricostruzione. Il vecchio campo `exact_reconstruction_rate` conserva la semantica
legacy: normalizza whitespace e può ignorare foglie non collocate. I nuovi report v2 aggiungono
`aligned_subset_reconstruction_normalized_rate`, `all_leaves_accounted_for_rate` e
`strict_full_reconstruction_rate`. Confrontare metriche con stessa versione; non rinominare
i risultati storici come strict. Nessuna di queste misure sostituisce la validazione dello schema
o il controllo umano delle categorie. I dettagli sono nella nota locale di implementazione R01–R02.

La copertura è solo un sostituto di quella proprietà: dice quali parole sono state collocate, mai in
che ordine. Un output che rimescolasse completamente le frasi otterrebbe comunque copertura piena.

Il comportamento di ciascuna metrica è bloccato da casi golden in `tests/metrics/cases/`, con i
punteggi attesi versionati accanto: una variazione non spiegata è un difetto, non un cambiamento
silenzioso di significato.

### Decoding vincolato

Impostando `inference.structured_output: true` (richiede l'extra `structured`) si richiede un
decoding vincolato allo schema. Compatibilità del backend, completamento entro il budget e
assenza di errori vanno verificati: **non è una garanzia incondizionata di JSON valido**.
Misurare comunque validità, troncamenti e fallimenti e distinguerli dalla qualità del contenuto.

In quella modalità la validità osservata è influenzata dal vincolo esterno e non misura da sola
quanto il modello abbia imparato il formato. Il valore sta nel confronto fra le due
esecuzioni — libera per misurare quanto il modello ha interiorizzato il formato, vincolata per
valutare la qualità del contenuto a parità di formato.

## Espansione del corpus

Il foglio consegnato proviene da conversazioni reali fra sviluppatori e assistente, dove le
richieste dirette dominano: i rami della tassonomia che descrivono il ragionamento vi comparivano
tre volte su 975 record, quindi tre foglie su nove non erano né imparabili né misurabili. Il comando
`synth` genera prompt aggiuntivi per quei paradigmi, partendo da trenta seed scritti a mano in
`data/seeds/paradigms.yaml`.

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

## Audit del corpus e valutazione downstream

Due strumenti restano fuori dal comando `pstparser` e si invocano come moduli, senza entry point
dedicati.

```bash
# Diagnostica in sola lettura: testo non coperto, corrispondenze tolleranti, possibili famiglie di duplicati
uv run python -m pstparser.data.audit --config configs/experiments/baseline.yaml --output outputs/new-run/corpus-audit.json
```

L'audit non corregge annotazioni né partizioni: i candidati (similarità di Jaccard sopra
`--similarity-threshold`, 0.9 di default) vanno rivisti a mano. Il file di output deve essere nuovo
e la sua directory deve già esistere.

```bash
uv run python -m pstparser.empirical render CANONICAL.jsonl VARIANTS.jsonl
uv run python -m pstparser.empirical generate VARIANTS.jsonl CONFIG.json GENERATION_DIR
uv run python -m pstparser.empirical check GENERATIONS.jsonl ORACLES.jsonl CHECK_DIR IMAGE@sha256:... ANALYSIS_ID
uv run python -m pstparser.empirical analyze DESIGN.json OUTCOMES.jsonl REPORT.json
```

`render` produce le varianti appaiate T/H-gold da annotazioni canoniche revisionate. `generate`
richiede una configurazione esplicita con revisione del modello immutabile e un record di revisione,
verifica il budget di token di tutte le varianti prima di generare e scrive un journal riprendibile
(`generations.jsonl`). `check` calibra gli oracoli funzionali forniti e li esegue in un container
Docker locale senza rete, con immagine fissata per digest. `analyze` esegue l'analisi appaiata sul
disegno dichiarato. Nessun comando sceglie da solo un benchmark: finché il protocollo non è
congelato, i loro output sono prove tecniche, non risultati.

Gli altri comandi downstream, nell'ordine in cui si usano (`--help` ne descrive gli argomenti):

| Fase | Comandi |
|---|---|
| Dati e campioni | `import-benchmark`, `calibrate-bigcodebench`, `merge-calibrations`, `sample`, `sample-pilot`, `sample-families` |
| Annotazione gold | `propose-spans` (solo bozze `draft`), `gold-lint`, `canonicalize`, `agreement` |
| Generazione | `render`, `generate` (anche su Colab con `notebooks/downstream.ipynb`) |
| Valutazione | `check-bigcodebench` (harness ufficiale, 2+1 esecuzioni), `calibrate-security`, `check-security` (CodeQL, un database per programma) |
| Pilot e freeze | `pilot-report` (non direzionale), `power`, `freeze`, `freeze-verify` |
| Analisi | `analyze`, `covariates`, `contamination`, `near-duplicates` |
| Tesi | `tables`, `figures`, `thesis-tables` |

`thesis-tables` rigenera ogni tabella (CSV e LaTeX) e figura (PDF e PNG) che i file di una run
consentono, in una directory nuova; ogni file riporta il `run_id`, lo SHA-256 di ogni input e il
digest dei sorgenti. Le figure richiedono l'extra `analysis` (matplotlib):

```bash
make thesis-tables DS_RUN=results/downstream/<run_id> TABLES=<directory nuova>
```

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
  empirical/     contratti downstream, renderer, oracoli e analisi appaiata
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

- **Il foglio consegnato non è quello usato nel lavoro originale.** Il notebook di partenza legge un
  file da circa 1208 righe che non è stato consegnato; questo repository ne contiene la versione da
  975 righe. I risultati numerici non sono quindi direttamente confrontabili con quelli pubblicati.
  Se il file originale venisse recuperato, basta aggiungerlo a `data.sources` in
  `configs/_base/data/`.
- **Il supporto del ramo `reasoning` è modesto e curato.** Le sue tre foglie contano 113, 86 e 33
  record su 1123, e provengono quasi per intero dai prompt annotati qui: il foglio consegnato ne
  attestava tre in tutto. I punteggi su quei nodi vanno letti con il loro support accanto, e i
  prompt che li esercitano sono stati generati e scelti perché li esercitassero — non sono un
  campione di come quei paradigmi compaiono in conversazioni reali.

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
