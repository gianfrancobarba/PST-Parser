# Risultati di riferimento

Questa directory ospita gli artefatti della run di riferimento, versionati insieme al codice:

| File | Contenuto |
|---|---|
| `predictions.jsonl` | Una predizione per riga, con il prompt e il target da cui deriva |
| `results.json` | Metriche aggregate |
| `eval_details.jsonl` | Punteggi per singolo esempio |
| `run_manifest.json` | Provenienza della run che ha prodotto le predizioni |

Il motivo per cui le predizioni sono versionate è che rendono le metriche **ricalcolabili senza
GPU e senza modello**:

```bash
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl
```

Chiunque può così rigenerare ogni numero in pochi secondi, verificarlo, o applicare metriche
diverse alle stesse predizioni.

La directory è al momento vuota: popolarla richiede una run di training e generazione su GPU con
il modello reale.
