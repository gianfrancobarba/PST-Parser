# Requisiti e limiti operativi

## Software

Il pacchetto dichiara Python `>=3.11,<3.13`; `.python-version` seleziona 3.11. Usare `uv` e
`uv.lock`, verificando le dipendenze risolte per sistema, interprete ed extra. Il lock non
rende identici kernel, driver e hardware. Docker e NVIDIA Container Toolkit servono al percorso
containerizzato con GPU; verificare driver/build CUDA con `nvidia-smi` e uno smoke del backend.
Git, quando disponibile, fornisce metadati in lettura: non occorrono staging, commit o push.

## Hardware e memoria

Preparazione, scoring PST e analisi appaiata non caricano modelli e possono girare su CPU.
La durata dipende dai record. Il codice generato downstream richiede un ambiente isolato
configurato separatamente: l'immagine di training non è automaticamente quel sandbox.

Training e generazione dipendono da modello, quantizzazione, batch, sequenze e backend.
La RTX 4070 da 8 GB del tesista **non garantisce la fattibilità della baseline**. Verificare
picco memoria e durata sui casi lunghi; 32 GB RAM non sostituiscono VRAM senza costi o limiti.
Eseguire parser e generatore separatamente.

Per ridurre il batch mantenendo il numero nominale di esempi per aggiornamento:

```text
--set training.per_device_train_batch_size=1 --set training.gradient_accumulation_steps=8
```

Questo **non garantisce equivalenza della traiettoria di training**: padding, token per batch,
riduzioni numeriche e implementazione possono cambiare. Registrare la configurazione effettiva.

Le vecchie misure di lunghezza su 1123 record descrivono uno snapshot. Dopo modifiche al corpus,
tokenizer, system prompt o chat template ricalcolare input, target e sequenze. Ridurre
`model.max_seq_length` può troncare dati; misurare quali e quanti. Una finestra ampia può
incidere sulla memoria a seconda del backend. `inference.max_new_tokens` è un limite,
non una garanzia di JSON completo. Spazio disco e download vanno misurati per pesi,
checkpoint, immagini e cache effettivi, senza assumere invarianti i vecchi valori in GB.

## Credenziali

`HF_TOKEN` serve quando richiesto dal repository del modello, eventualmente dopo accettazione
delle condizioni di accesso. `NVIDIA_API_KEY` riguarda il provider sintetico configurato.
Copiare `.env.example` in `.env` e compilare solo ciò che serve. Non includere credenziali
negli artefatti condivisi.

## Riproducibilità e stato scientifico

Le trasformazioni senza modello sono ripetibili a parità di input, configurazione e ambiente
compatibile. Seed e greedy decoding **non garantiscono determinismo bit per bit** tra versioni,
hardware o backend. Il manifest registra seed e impostazioni; non è un rilevatore completo
di ogni operazione non deterministica. Hash e manifest per stadio migliorano tracciabilità,
senza garantire la recuperabilità di ogni artefatto esterno.

I moduli downstream sono preparazione tecnica: il docente non ha ancora approvato il protocollo
e non sono risultati dell'esperimento T/H. Gli output storici non provano l'esecuzione della
configurazione corrente. Seguire [INSTALL.md](INSTALL.md) per directory fresche e split separati.

Vincoli Colab, hardware e fonti primarie dei modelli sono descritti nella nota di fattibilità
della documentazione locale di progettazione, non versionata.
Il canone dichiarato non garantisce una GPU specifica né ore illimitate.
