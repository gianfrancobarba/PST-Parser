# Requisiti

## Software

| Componente | Versione | Note |
|---|---|---|
| Python | 3.11 o 3.12 | Gestito da `uv`, non serve installarlo a mano |
| [`uv`](https://docs.astral.sh/uv/) | 0.11 o successiva | Gestore di ambiente e dipendenze |
| Git | qualsiasi | Il manifest di run registra il commit corrente |
| Docker | 24 o successiva | Solo per l'esecuzione containerizzata |
| NVIDIA Container Toolkit | 1.17 o successiva | Solo per il training in container |

Le versioni esatte di tutte le dipendenze Python sono fissate in `uv.lock`, che è versionato.

## Hardware

Il progetto ha due profili di esecuzione con requisiti molto diversi.

### Calcolo delle metriche

Nessun requisito particolare: qualsiasi macchina con Docker o Python. Il comando `score` non carica
alcun modello e non importa né PyTorch né una libreria di modelli. Su un file di predizioni da
qualche centinaio di righe impiega pochi secondi.

### Training e generazione

| Risorsa | Minimo | Note |
|---|---|---|
| GPU | NVIDIA con 16 GB di VRAM | Il modello base è caricato a 4 bit |
| Driver NVIDIA | 550 o successivo | Richiesto da CUDA 12.8 |
| RAM di sistema | 16 GB | |
| Spazio su disco | 40 GB | Immagine, pesi del modello base, artefatti delle run |

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
