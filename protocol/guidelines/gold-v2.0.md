# Linee guida del gold downstream — `gold-v2.0`

Copia congelata delle regole con cui è stato rivisto il gold di BigCodeBench-Instruct (prefisso di 600 task, `data/annotated/downstream/bcb-gold/`) e di LLMSecEval S2 (100 prompt, `data/annotated/llmseceval-gpl/lse-final/`), e dei rispettivi piloti. Testo in italiano, come l'originale; trascritto a mano il 19/11/2026. `gold-v2.0` è il valore di `annotation_version` e di `guidelines_version` su ogni riga.

**Che cosa non ne fa parte.** Il 18/11/2026 era stato proposto un addendum di sette convenzioni (A56). Le unità decise con quelle convenzioni sono state riportate alla foglia coerente con `gold-v2.0`, su decisione dello studente: 22 in BigCodeBench (prefisso), 13 in LLMSecEval, 7 e 4 nei due piloti. I registri sono `review-log.csv` (colonna `ripristino_a56`) e `provenance.json` accanto a ogni gold. Il gold congelato segue soltanto le regole di questo file.

## 1. Oggetto e contratto (invariati dalla versione 1.0)

- Si annota soltanto il **messaggio canonico** del benchmark (il campo sorgente congelato con hash), mai il wrapper di istruzione né il system. Non si usano soluzioni, test, CWE attesa né output di un modello.
- **Span**: offset in caratteri Unicode, `start` incluso ed `end` escluso; `phrase` è esattamente `text[start:end]`. Gli span non si sovrappongono; il testo non coperto è solo spazio bianco, assegnato dal renderer (gli spazi fra due segmenti e il suffisso vanno al segmento precedente, il prefisso al primo). Nessun carattere non bianco resta fuori.
- **Foglie** (nove, ordine tassonomico):

| Foglia | Criterio operativo |
|---|---|
| `main_instruction` | Azione o risultato richiesto al modello |
| `context.format` | Forma positiva della risposta: layout, struttura, stile |
| `context.constraints` | Restrizioni, esclusioni, limiti e requisiti tecnici obbligatori |
| `context.data` | Materiale dato da elaborare: codice, testo, log, scenario descrittivo |
| `context.role` | Persona assegnata al modello, non ruolo dell'utente descritto |
| `examples` | Dimostrazioni input/output senza passaggi di ragionamento espliciti |
| `reasoning.influence` | Istruzioni sul procedimento di ragionamento |
| `reasoning.reasoning_examples` | Dimostrazioni con input, ragionamento e output espliciti |
| `reasoning.paths` | Percorsi o alternative già esplicitati nel testo |

- Non si introducono segmenti per riempire categorie assenti. Stati di una riga: `draft` → `reviewed` → `frozen`.

## 2. Regole generali

- **G1 — Unità.** Lo span minimo è una sezione a formula fissa oppure una **frase intera**. Non si spezza una frase. Confini di frase: `[.!?]` seguito da spazio; su LLMSecEval anche `.` seguito da maiuscola senza spazio ("loaded.The"). Non sono confini le abbreviazioni (`e.g.`, `i.e.`, `etc.`), i decimali, i punti dentro identificatori o codice (`os.path.join`, `matplotlib.axes.Axes`).
- **G2 — Frasi miste.** Se una frase contiene sia azione sia restrizione, la foglia è quella del suo ruolo prevalente secondo l'albero G3; il caso si registra come `mixed`.
- **G3 — Albero di decisione per frase** (fuori dalle sezioni a formula):
  1. prima frase della descrizione (in LLMSecEval la prima dopo il prefisso) → `main_instruction`;
  2. dimostrazione input/output (`>>>`, "Example N:") → `examples`;
  3. restrizione o obbligo: *only, must, must not, never, do not, should not, cannot, always, at most, at least, exactly, without, ensure*, gestione di errori o input invalidi, tecnologia o libreria imposta ("using X", "uses X") → `context.constraints`;
  4. sola descrizione della forma del risultato (tipo e struttura del valore restituito, titolo, etichette, colori del grafico, colonne, formato di stampa) → `context.format`;
  5. fatti forniti sull'ambiente (costanti con valori, file o funzioni già esistenti da usare, descrizione dei parametri) → `context.data`;
  6. altrimenti (descrizione del comportamento da implementare) → `main_instruction`.
- **G4 — Foglie che restano vuote salvo evidenza testuale**: `context.role` (assegnazione di una persona al modello; "You are given a list…" descrive l'input e segue G3) e `reasoning.*`.

## 3. Casi di confine

| # | Caso | Regola |
|---|---|---|
| B1 | "The function should raise the exception for: …" | Intera sezione, formula inclusa, **un** span `context.constraints` |
| B2 | "The function should output with: …" | Intera sezione, **un** span `context.format`, anche con clausole condizionali ("Returns None if…") |
| B3 | "You should write self-contained code starting with:" | `context.constraints` (restrizione e forma inseparabili: priorità a constraints) |
| B4 | Codice iniziale | Dalla ```` ``` ```` di apertura a quella di chiusura, **un** span `context.data`, mai spezzato |
| B5 | Import e librerie | Restano nel blocco `context.data`. Una libreria nominata in una frase non iniziale ("It uses pandas…") → `context.constraints` per G3.3; nella prima frase resta `main_instruction` (G1) |
| B6 | "Note that: Notes: …" | Ogni frase uno span. Default `context.constraints`; `context.format` solo se descrive la forma (per esempio un titolo del grafico). "Note that: Notes:" fa parte della prima frase |
| B7 | Doctest nella descrizione (`>>> …`, preceduto da "Example N:") | Un span `examples` fino alla sezione successiva |
| B8 | `Args:`, `Parameters:`, `Constants:` nella descrizione | Un span `context.data` |
| B9 | Frasi successive alla prima, descrittive del comportamento ("It then opens and reads this file…") | `main_instruction` |
| B10 | LLMSecEval: "Generate Python code for the following:" | `main_instruction` |
| B11 | LLMSecEval: inquadramento "The code is a simple web application that…" | `main_instruction` |
| B12 | LLMSecEval: frasi che citano codice inline | Frase intera, foglia per G3; il frammento non si separa |
| B13 | Testo irregolare (".The", doppi spazi, `<url>`) | Nessuna correzione; G1 per i confini |
| B14 | Caratteri invisibili (U+200B) | Sono contenuto: inclusi nello span che li contiene |

## 4. Esempi dal gold congelato

- `BigCodeBench/39`: "Calculate the mean value … significantly." → `main_instruction` (G3.1); "- Create a lineplot with the mean of rows in red." → `main_instruction` (G3.6); "Its label is 'Means'." e "This plot should be blue." → `context.format` (G3.4); "Example 2: >>> data = …" → `examples` (B7); sezione "The function should output with:" → `context.format` (B2); "You should write self-contained code starting with:" → `context.constraints` (B3); codice iniziale → `context.data` (B4).
- `LSE-071`: "Generate Python code for the following:" → `main_instruction` (B10); "The p.save(os.path.join('uploads' filename)) line saves the file to the uploads folder." → `main_instruction` (B12, G3.6); "The uploads folder is created in the same directory as the source file." → `main_instruction` (G3.6).

## 5. Procedura con cui il gold è stato prodotto (dichiarata)

1. **Bozze a regole fisse**: `rule:bcb-markers-v1` (BigCodeBench) e `rule:lse-prefix-v1` (LLMSecEval), nel campo `annotator`. Confini e foglie delle sezioni a formula, della prima frase e del codice sono fissati dalle regole.
2. **Unità di giudizio** (`judgement-units-v2`): le frasi della prosa dopo la prima (G3) e le frasi di "Note that:" (B6). Sono le sole unità la cui foglia è decisa dal revisore.
3. **Proposte di un LLM** (GPT, OpenAI; versione non registrata) sulla foglia delle sole unità di giudizio. All'LLM non sono stati dati soluzioni, test, output o predizioni del parser.
4. **Decisione dello studente** (`reviewer: studente`) su ogni unità, senza spezzare né unire span. Il registro è `review-log.csv` (proposta, decisione, ripristino A56). Le decisioni finali coincidono con la proposta dell'LLM in 1.205 unità su 1.279. Tempi non misurati.
5. **Freeze**: lo stato `frozen` è impostato dallo studente; `frozen-record.json` registra gli hash prima e dopo.
6. **Misura indipendente**: ri-annotazione da zero, senza assistenza, di 30 task BigCodeBench e 30 prompt LLMSecEval (campioni in `protocol/reannotation/`), con κ sulle unità di giudizio.
