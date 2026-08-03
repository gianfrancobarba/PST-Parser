# Requisiti

## Software

| Componente | Versione | Note |
|---|---|---|
| Python | 3.11 | Fissata in `.python-version`; `uv` la installa da sé |
| [`uv`](https://docs.astral.sh/uv/) | 0.11 o successiva | Gestore di ambiente e dipendenze |
| Git | qualsiasi | Il manifest di run registra il commit corrente |
| Docker | 24 o successiva | Solo per l'esecuzione containerizzata |
| NVIDIA Container Toolkit | 1.17 o successiva | Solo per il training in container |

Le versioni esatte di tutte le dipendenze Python sono fissate in `uv.lock`, che è versionato.

Il pacchetto dichiara di supportare sia 3.11 sia 3.12, ma l'ambiente del repository è fissato a
**3.11** in `.python-version`. Senza quel vincolo `uv` sceglierebbe l'interprete più alto
disponibile, e il lock risolve dipendenze diverse a seconda della versione: lo stesso `uv sync`
produrrebbe ambienti diversi su macchine diverse. La versione fissata è anche quella su cui sono
tarati il controllo dei tipi, le regole di stile e le immagini container.

## Hardware

Il progetto ha due profili di esecuzione con requisiti molto diversi.

### Calcolo delle metriche

Nessun requisito particolare: qualsiasi macchina con Docker o Python. Il comando `score` non carica
alcun modello e non importa né PyTorch né una libreria di modelli. Su un file di predizioni da
qualche centinaio di righe impiega pochi secondi.

### Training e generazione

| Risorsa | Minimo | Note |
|---|---|---|
| GPU | NVIDIA con 8 GB di VRAM | Il modello base è caricato a 4 bit |
| Driver NVIDIA | 550 o successivo | Richiesto da CUDA 12.8 |
| RAM di sistema | 16 GB | |
| Spazio su disco | 40 GB | Immagine, pesi del modello base, artefatti delle run |

#### GPU da 8 GB

La configurazione di riferimento usa `per_device_train_batch_size: 2`, che su 8 GB di VRAM è al
limite. La via corretta è **dimezzare il batch e raddoppiare l'accumulo**:

```bash
--set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8
```

La dimensione effettiva del batch resta 8, quindi la traiettoria di ottimizzazione è la stessa:
cambia solo il picco di memoria per le attivazioni.

Sul corpus disponibile le sequenze di training, comprensive del system prompt da 542 token, hanno
questa distribuzione:

| | token |
|---|---:|
| mediana | 836 |
| 75° percentile | 1372 |
| 90° percentile | 2162 |
| 95° percentile | 2646 |
| 99° percentile | 3415 |
| massimo | 5314 |

Solo **4 sequenze su 975 (0.4%)** superano i 4096 token e vengono troncate. La memoria occupata
dalle attivazioni dipende dalla lunghezza effettiva del batch, non da `max_seq_length`: dato che
metà degli esempi sta sotto gli 836 token, la maggior parte dei passi è ben al di sotto del caso
peggiore.

Se anche a batch 1 la memoria non basta, ridurre `model.max_seq_length` a 3072 tronca l'1.7% degli
esempi invece dello 0.4%.

Il modello base viene scaricato al primo utilizzo (circa 6 GB a 4 bit) e conservato nella cache di
Hugging Face. In Docker la cache è un volume nominato, quindi sopravvive alla ricostruzione delle
immagini.

## Credenziali

| Variabile | Necessaria per | Come ottenerla |
|---|---|---|
| `HF_TOKEN` | Scaricare il modello base | https://huggingface.co/settings/tokens, permesso di lettura |
| `NVIDIA_API_KEY` | Generazione di prompt sintetici | Solo se si usa quel comando |

Copiare `.env.example` in `.env` e compilarlo. Il file `.env` non è versionato.

Alcuni modelli base sono ad accesso condizionato: occorre accettarne la licenza sul proprio account
Hugging Face prima che il download funzioni.

## Determinismo

Sono deterministici, a parità di configurazione: la costruzione del corpus, la partizione, l'ordine
degli esempi, il templating e l'intera pipeline di valutazione, che decodifica in modo greedy.

Il training **non** è riproducibile bit per bit. Con adapter a rango ridotto in bfloat16 l'ordine
delle riduzioni non è garantito e i kernel di attenzione ottimizzati non hanno una versione
deterministica del passo all'indietro. I seed applicati e le operazioni non deterministiche
rilevate sono registrati nel manifest di ogni run.
