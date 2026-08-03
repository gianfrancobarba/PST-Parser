# Risultati di riferimento

Questa directory ospita gli artefatti della run di riferimento, versionati insieme al codice:

| File | Contenuto |
|---|---|
| `predictions.jsonl` | Una predizione per riga, con il prompt e il target da cui deriva |
| `alignments.jsonl` | Le stesse predizioni e i loro riferimenti, con ogni frase collocata nel prompt |
| `results.json` | Metriche aggregate, come le calcolava il lavoro di partenza |
| `eval_details.jsonl` | Punteggi per singolo esempio, idem |
| `conformant/` | Le stesse predizioni, ripunteggiate secondo le definizioni pubblicate |
| `run_manifest.json` | Provenienza della generazione che ha prodotto le predizioni |
| `train_manifest.json` | Provenienza del training che ha prodotto l'adapter |
| `training_metrics.jsonl` | Curva di loss, un record per ogni punto registrato |

Il motivo per cui le predizioni sono versionate è che rendono le metriche **ricalcolabili senza
GPU e senza modello**:

```bash
uv run pstparser score --config configs/experiments/baseline.yaml --predictions results/predictions.jsonl --run-dir results/conformant
```

Chiunque può così rigenerare ogni numero in pochi secondi, verificarlo, o applicare metriche
diverse alle stesse predizioni.

> **`results.json` ed `eval_details.jsonl` non vanno rigenerati.** Sono l'output del codice al tag
> `baseline-as-is` e servono da termine di paragone; il codice corrente calcola le metriche in modo
> diverso e li sovrascriverebbe. Per riprodurli serve quel tag:
> `git worktree add ../pst-asis baseline-as-is`. Ogni nuovo punteggio va scritto in una
> sottodirectory, con `--run-dir`.

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

## Metriche

**Nessuna predizione è cambiata fra le due colonne.** Cambiano solo le definizioni con cui sono
misurate: la colonna di sinistra riproduce il calcolo del lavoro di partenza, quella di destra
applica le formule come sono pubblicate.

| Metrica | Come calcolata in origine | Secondo le definizioni |
|---|---:|---:|
| Validità JSON | 93.18% | 93.18% |
| Coverage score | 99.71% | 99.60% |
| Hallucination rate | 0.39% | 0.41% |
| Tree edit distance | 1.55 | 1.55 |

| Foglia | In origine | Secondo le definizioni | Support |
|---|---:|---:|---:|
| `context.role` | 1.0000 | 1.0000 | 24 |
| `main_instruction` | 0.9127 | 0.8806 | 1294 |
| `context.data` | 0.7974 | **0.9070** | 3268 |
| `examples` | 0.7775 | 0.7021 | 198 |
| `context.format` | 0.5238 | **0.8105** | 668 |
| `context.constrains` | 0.4830 | **0.7097** | 585 |
| `reasoning.influence` | — | n/d | 0 |
| `reasoning.reasoning_examples` | — | n/d | 0 |
| `reasoning.paths` | — | n/d | 0 |

### Cosa è cambiato, e perché

**Copertura e allucinazione: quasi nulla.** Le due formule confrontano insiemi di token su entrambi
i lati della frazione; il calcolo di partenza ne trattava uno come lista, e in modo opposto nelle
due metriche. Correggerlo sposta i valori di un decimo di punto. È una correzione di conformità,
non un ribaltamento di risultati, e vale la pena dirlo esplicitamente perché il contrario sarebbe
stato facile da suggerire.

**F1 per campo: molto.** `context.format` sale di 0.29 e `context.constrains` di 0.23. La causa è
che i segmenti venivano appaiati **per posizione nella lista**: se il modello produceva due segmenti
dove l'annotatore ne aveva registrato uno, il primo veniva confrontato con il primo e il secondo con
la stringa vuota, penalizzando due volte un contenuto corretto. Aggregando i token del campo prima
del confronto, come la definizione prescrive, quella penalità sparisce. **La bassa concordanza sui
due campi "difficili" era in larga parte un artefatto dell'appaiamento, non una debolezza del
modello** — il che cambia la lettura di quale sia il problema aperto su quei nodi.

**Il support ora accompagna ogni punteggio.** Un 1.0000 su 24 token e un 0.9070 su 3268 non sono la
stessa affermazione, e le tre foglie del ramo `reasoning` risultano esplicitamente non valutabili
invece di essere assenti dalla tabella.

**La validità è più severa di prima e dà lo stesso numero.** Ora una predizione conta come valida
se si parsifica **e** rispetta lo schema derivato dalla tassonomia: un `{}` non passa più. Tutte e
82 le predizioni parsabili sono anche conformi, quindi il valore non si muove. La coincidenza è un
risultato, non una svista.

## Ricostruzione del prompt

È la prima misurazione della proprietà su cui il framework è argomentato: raccogliere le foglie,
ordinarle per posizione e ritrovare il prompt di partenza. Il coverage score ne era solo un
sostituto, perché riduce la predizione a un sacchetto di token e dice **quali** parole sono state
collocate, mai **in che ordine**.

| | Predizione | Riferimento |
|---|---:|---:|
| Frasi | 174 | 194 |
| Collocate nel prompt | 88.51% | 88.14% |
| Con più di un'occorrenza candidata | 0.00% | 0.00% |
| **Ricostruiscono il prompt** | **72.73%** | 79.55% |
| ... fra quelle parsificabili | 78.05% | 79.55% |

La ricostruzione è confrontata a meno di ogni carattere di spaziatura, perché lo spazio fra due
segmenti non appartiene a nessuno dei due.

**Il divario fra 99.60% di copertura e 72.73% di ricostruzione è il contributo di questa misura.**
Dice che quasi tutte le parole del prompt finiscono da qualche parte, ma che in un caso su quattro
l'insieme delle foglie non basta a rimettere insieme il testo. Su 88 predizioni: 6 non si
parsificano, e 18 lasciano scoperta una porzione del prompt — dalla frase iniziale mai segmentata
(indice 869, 45 caratteri) alla subordinata finale omessa (indice 136, 77 caratteri).

**Il riferimento non ricostruisce meglio del modello di molto.** Il 12% di frasi annotate non
collocabili non è un difetto del modello: è il ground truth che non si ritrova nel proprio prompt.
Misurare i due lati separatamente è ciò che impedisce di addebitare al modello un difetto del
formato di annotazione, ed è la ragione per cui `alignments.jsonl` riporta entrambi.

**L'ambiguità è zero, e va comunque guardata a ogni run.** Nessuna frase, né predetta né annotata,
ha più di un'occorrenza candidata: la regola di disambiguazione non è mai stata chiamata a
scegliere. È un dato su questo corpus, non un teorema — segmenti più corti la farebbero salire.

## Dove finiscono i token del prompt

La matrice di confusione è costruita sui token del prompt, non sui valori di campo interi:
entrambi gli alberi vengono collocati nel prompt, e per ogni token si confronta l'etichetta del
riferimento con quella della predizione. `none` è il token che un lato non ha assegnato ad alcun
nodo.

| Atteso | Predetto | Token |
|---|---|---:|
| `context.data` | `none` | 622 |
| `none` | `examples` | 473 |
| `none` | `context.data` | 200 |
| `none` | `main_instruction` | 174 |
| `context.constrains` | `context.format` | 117 |

Le prime quattro righe sono errori di **confine**, non di categoria: testo che un lato colloca e
l'altro no. Il calcolo precedente confrontava valori di campo interi e scartava in silenzio ciò che
non combaciava alla lettera, quindi non poteva vederne nemmeno uno. La quinta riga è invece la
confusione semantica fra istruzioni presentazionali e restrittive, ora quantificata in token.

## Ablazione: la punteggiatura

Le metriche di contenuto normalizzano via la punteggiatura, quindi non misurano mai se un segmento
è stato tagliato una virgola troppo presto. Ricalcolando le stesse 88 predizioni con una
tokenizzazione che tiene ogni segno come token a sé:

| Metrica | Punteggiatura scartata | Punteggiatura tenuta |
|---|---:|---:|
| Coverage score | 99.60% | 99.54% |
| Hallucination rate | 0.41% | 0.61% |
| F1 `context.constrains` | 0.7097 | 0.7101 |
| F1 `context.data` | 0.9070 | 0.9148 |
| F1 `context.format` | 0.8105 | 0.8216 |
| F1 `examples` | 0.7021 | 0.7583 |
| F1 `main_instruction` | 0.8806 | 0.8837 |

Lo scarto è piccolo ovunque tranne che sull'allucinazione, che cresce di due decimi di punto. Il
default resta invariato: cambiarlo insieme alle definizioni avrebbe reso impossibile attribuire i
delta all'uno o all'altro, e la tabella dice che non c'è molto da guadagnarci. Se un giorno lo si
cambierà, sarà una decisione motivata da questi numeri e non da un'impressione.

## Come vanno lette

**I due insiemi di valutazione sono diversi** da quelli del lavoro pubblicato: 88 esempi qui contro
109, estratti da corpora di dimensioni diverse. I confronti sono indicativi, non appaiati.

**Tre foglie su nove non hanno supporto.** `reasoning.influence` ha tre esempi annotati in tutto,
`reasoning.reasoning_examples` e `reasoning.paths` nessuno: non c'è nulla da misurare.

**La validità JSON non raggiunge il 100%.** Sei predizioni su 88 non sono parsificabili. Il budget
di generazione e la finestra di contesto coincidono, quindi i prompt con un campo `data` esteso
possono esaurire lo spazio prima della chiusura dell'oggetto. La generazione vincolata a schema
(`inference.structured_output`) elimina il problema per costruzione, al prezzo di rendere la
metrica non più informativa: per questo resta disattivata in questa run e il confronto fra le due
esecuzioni è più interessante di ciascuna presa da sola.

## Tentativi

`attempts/` raccoglie i resoconti delle esecuzioni non andate a buon fine, con i registri completi.
Servono a non ripetere diagnosi già fatte e a documentare i vincoli hardware incontrati.

| Data | Hardware | Esito |
|---|---|---|
| 2026-08-01 | RTX 4070 Laptop, 8 GB | interrotto allo step 25 su 300 per throttling termico; nessun modello prodotto |
