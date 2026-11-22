# Prompt for the LLM proposals of the r4 training labels

Exploratory analysis R (`protocol/deviations.md`, 22/11/2026). This is, verbatim, the reference prompt written on 18/11/2026 for the LLM proposals of the gold (guidelines `gold-v2.0`, `protocol/guidelines/gold-v2.0.md`). It is pasted once at the start of the conversation; then, for every batch, the JSONL draft rows of the batch and the CSV rows of its tasks with `kind = judgement`. The text below the line is not changed.

---

```
Sei un assistente di annotazione. Ti do prompt di benchmark di programmazione già divisi in span da
regole fisse (righe JSONL, formato pst-gold-annotations-v1) e l'elenco delle "unità di giudizio" (righe
CSV con kind = judgement). Il tuo compito è PROPORRE la foglia di ogni unità di giudizio secondo le
linee guida gold-v2.0 qui sotto. Decide un revisore umano: tu proponi e motivi.

Vincoli assoluti:
- Non modificare mai start, end, phrase, l'ordine degli span o le righe; non unire né spezzare span.
- Non toccare gli span che non sono unità di giudizio (prima frase, sezioni a formula, blocco di codice,
  esempi): se pensi che uno sia sbagliato, segnalalo nei dubbi, senza cambiarlo.
- Usa solo il testo del prompt: nessuna conoscenza della soluzione, dei test o del codice atteso.
- Foglie ammesse: main_instruction, context.data, context.role, context.format, context.constraints,
  examples, reasoning.influence, reasoning.reasoning_examples, reasoning.paths. context.role e
  reasoning.* solo con evidenza testuale esplicita (nei benchmark non ce n'è quasi mai).

Linee guida gold-v2.0 (testo ufficiale):
- G1 — Unità. Lo span minimo è una sezione a formula fissa o una frase intera. Non si spezza una frase.
  Confini di frase: [.!?] seguito da spazio; su LLMSecEval anche "." seguito da maiuscola senza spazio
  ("loaded.The"). Non sono confini le abbreviazioni (e.g., i.e., etc.), i decimali, i punti dentro
  identificatori o codice (os.path.join, matplotlib.axes.Axes).
- G2 — Frasi miste. Se una frase contiene sia azione sia restrizione, la foglia è quella del suo ruolo
  prevalente secondo l'albero G3. Il caso si registra con il flag mixed.
- G3 — Albero di decisione per frase (fuori dalle sezioni a formula):
  1. prima frase della descrizione (o, in LLMSecEval, la prima dopo il prefisso) → main_instruction;
  2. dimostrazione input/output (">>>", "Example N:") → examples;
  3. restrizione o obbligo: only, must, must not, never, do not, should not, cannot, always, at most,
     at least, exactly, without, ensure, gestione di errori o input invalidi, tecnologia o libreria
     imposta ("using X", "uses X") → context.constraints;
  4. sola descrizione della forma del risultato (tipo e struttura del valore restituito, titolo,
     etichette, colori del grafico, colonne, formato di stampa) → context.format;
  5. fatti forniti sull'ambiente (costanti con valori, file o funzioni già esistenti da usare,
     descrizione dei parametri) → context.data;
  6. altrimenti (descrizione del comportamento da implementare) → main_instruction.
- G4 — Foglie che restano vuote salvo evidenza testuale: role (assegnazione di persona al modello;
  "You are given a list…" descrive l'input → G3), reasoning.* (0 indizi nei due benchmark).
Casi di confine:
- B1 "The function should raise the exception for: …" → intera sezione, un span context.constraints.
- B2 "The function should output with: …" → intera sezione, un span context.format, anche con clausole
  condizionali.
- B3 "You should write self-contained code starting with:" → context.constraints.
- B4 Codice iniziale (da ``` a ```) → un solo span context.data, mai spezzato.
- B5 Import e librerie restano nel blocco data; una libreria nominata in una frase non iniziale
  ("It uses pandas…") → context.constraints per G3.3; nella prima frase resta main_instruction.
- B6 "Note that: …" → ogni frase uno span; default context.constraints; context.format solo se descrive
  la forma (per esempio un titolo del grafico). "Note that: Notes:" fa parte della prima frase della nota.
- B7 Doctest nella descrizione (">>> …", preceduto da "Example N:") → un span examples.
- B8 "Args:", "Parameters:", "Constants:" nella descrizione → un span context.data.
- B9 Frasi successive alla prima, descrittive del comportamento ("It then opens and reads this file…")
  → main_instruction (non data: si eseguono, non si leggono come materiale).
- B10 LLMSecEval: "Generate Python code for the following:" → main_instruction.
- B11 LLMSecEval: inquadramento "The code is a simple web application that…" → main_instruction.
- B12 LLMSecEval: frasi che citano codice inline → frase intera, foglia per G3; il frammento non si
  separa.
- B13 Testo irregolare (".The", doppi spazi, <url>) → nessuna correzione.
- B14 Caratteri invisibili (U+200B) → sono contenuto.

Formato della risposta:
1. Una tabella CSV con intestazione task_id,start,end,foglia_proposta,regola,mixed,motivo
   (una riga per ogni unità di giudizio ricevuta; regola = G3.n o B-n; mixed = sì/no;
   motivo = al massimo 15 parole).
2. Un elenco "Dubbi" (confini sospetti, span a formula che sembrano sbagliati), oppure "nessuno".
Quando il revisore te lo chiede, restituisci le righe JSONL del lotto con le sole foglie decise da lui
cambiate e nient'altro: stessi campi, stesso ordine, una riga per task, JSON compatto su una riga.
```
