# Risultati di riferimento

Questa directory ospita gli artefatti della run di riferimento, versionati insieme al codice:

| File | Contenuto |
|---|---|
| `predictions.jsonl` | Una predizione per riga, con il prompt e il target da cui deriva |
| `results.json` | Metriche aggregate |
| `eval_details.jsonl` | Punteggi per singolo esempio |
| `run_manifest.json` | Provenienza della generazione che ha prodotto le predizioni |
| `train_manifest.json` | Provenienza del training che ha prodotto l'adapter |
| `training_metrics.jsonl` | Curva di loss, un record per ogni punto registrato |

Il motivo per cui le predizioni sono versionate è che rendono le metriche **ricalcolabili senza
GPU e senza modello**:

```bash
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

Chiunque può così rigenerare ogni numero in pochi secondi, verificarlo, o applicare metriche
diverse alle stesse predizioni.

I pesi dell'adapter non sono versionati: pesano 167 MB e non servono a riprodurre alcuna metrica.
Servono solo a generare nuove predizioni, e per quello basta rieseguire il training a partire dal
manifest.

## La run

| | |
|---|---|
| Data | 2 agosto 2026 |
| Hardware | NVIDIA L4, 24 GB (Google Colab) |
| Configurazione | `configs/experiments/baseline.yaml`, senza sovrascritture |
| Training | 300 step in 58 minuti, batch effettivo 8 |
| Generazione | 88 predizioni in 33 minuti, decodifica greedy |
| Corpus | 975 record, split 887 / 88 |

### Curva di loss

| Step | Training | Validazione |
|---:|---:|---:|
| 50 | 0.2906 | 0.2922 |
| 100 | 0.3171 | 0.2736 |
| 150 | 0.2477 | 0.2683 |
| 200 | 0.2252 | **0.2622** |
| 250 | 0.1686 | 0.2644 |
| 300 | 0.2076 | 0.2660 |

La loss di validazione tocca il minimo allo step 200 e risale nei cento step successivi. Poiché
`load_best_model_at_end` è disattivato, l'adapter salvato è quello dello step 300: non il migliore
dei sei valutati. È il comportamento del lavoro di riferimento, riprodotto fedelmente, e la curva
qui sopra è ciò che permette di constatarlo.

### Metriche

| Metrica | Questa run | Lavoro di riferimento |
|---|---:|---:|
| Validità JSON | 93.18% | 100% |
| Coverage score | 99.71% | 99.91% |
| Hallucination rate | 0.39% | 0.18% |
| Tree edit distance | 1.55 | non riportata |

| Foglia | F1 | Lavoro di riferimento |
|---|---:|---:|
| `context.role` | 1.0000 | 0.7857 |
| `main_instruction` | 0.9127 | 0.8662 |
| `context.data` | 0.7974 | 0.8805 |
| `examples` | 0.7775 | 1.0000 |
| `context.format` | 0.5238 | 0.5928 |
| `context.constrains` | 0.4830 | 0.3453 |

### Come vanno lette

**I due insiemi di valutazione sono diversi**: 88 esempi qui contro 109, estratti da corpora di
dimensioni diverse. I confronti sono indicativi, non appaiati.

**Tre foglie su nove non compaiono.** `reasoning.influence` ha tre esempi annotati in tutto,
`reasoning.reasoning_examples` e `reasoning.paths` nessuno: non c'è nulla da misurare. La stessa
assenza caratterizza il lavoro di riferimento.

**La validità JSON non raggiunge il 100%.** Sei predizioni su 88 non sono parsificabili. Il budget
di generazione e la finestra di contesto coincidono, quindi i prompt con un campo `data` esteso
possono esaurire lo spazio prima della chiusura dell'oggetto. La generazione vincolata a schema
(`inference.structured_output`) elimina il problema per costruzione, al prezzo di rendere la
metrica non più informativa: per questo resta disattivata in questa run e il confronto fra le due
esecuzioni è più interessante di ciascuna presa da sola.

**La loss di validazione non è confrontabile con quella riportata dal lavoro originale** (circa
0.6). La loss di training lo è, e coincide; se le curve di training combaciano e quelle di
validazione no, la differenza sta nei dati di valutazione, non nella ricetta. Il corpus disponibile
è più piccolo di quello usato in origine e non contiene i prompt sintetici dedicati ai paradigmi di
ragionamento, che erano gli esempi più vari.

## Tentativi

`attempts/` raccoglie i resoconti delle esecuzioni non andate a buon fine, con i registri completi.
Servono a non ripetere diagnosi già fatte e a documentare i vincoli hardware incontrati.

| Data | Hardware | Esito |
|---|---|---|
| 2026-08-01 | RTX 4070 Laptop, 8 GB | interrotto allo step 25 su 300 per throttling termico; nessun modello prodotto |
