# Studio esperti — materiali (proposta)

Specifica e criteri: `docs/knowledge/testing/expert-study.md`.
Report della proposta iniziale (3 task): `report/expert-study-report.pdf`.
Pilota del 20/11 (grafici e progetti del collega): `report/pilot/pilot-report.pdf`.

Due task: T1 MLP (`task1-mlp`), T2 attention multi-testa con metavariabili
(`task2-multihead`). Nel T2 il foglio chiede di finire con `d-model`=64 e
`num-heads`=8; chi lascia 32/4 non viene bocciato.

| Cartella | Contenuto |
| --- | --- |
| `tasks/expert-tasks.tex` | Foglio unico dei task (inglese) per NNModelling e PyTorch |
| `projects/<task>/start` | Progetto NNModelling da cui parte l'esperto |
| `projects/<task>/solution` | Soluzione di riferimento (solo per il facilitatore) |
| `notebooks/` | Notebook Colab dei partecipanti (generati); `solutions/` per il facilitatore |
| `report/` | Report LaTeX e screenshot |

Strumenti in `tools/expert_study/`: `study_checks.py` (controlli PyTorch,
copiato accanto ai notebook), `grade.py` (correzione automatica di progetti e
notebook), `prepare_usb.py` (pennetta), `split_activity_log.py` e
`analyse_activity.py` (tempi per task dal log di NNModelling), `plot_session.py`
(grafici di un partecipante; serve matplotlib).

## Notebook su Google Colab

I task PyTorch si fanno su Colab dal browser, con l'account Google dell'esperto:

- T1: <https://colab.research.google.com/github/LucaSforza/nnmodelling-ui/blob/expert-study-materials/study/experts/notebooks/task1_mlp.ipynb>
- T2: <https://colab.research.google.com/github/LucaSforza/nnmodelling-ui/blob/expert-study-materials/study/experts/notebooks/task2_multihead.ipynb>

Struttura (protocollo in `docs/knowledge/testing/expert-study.md`):

1. **Step 1**, cella visibile col codice nascosto: ID partecipante, poi ▶.
   Scarica `study_checks.py` e i dati e fa partire il timer (evento `start`).
2. **Your model**, **Try it**, **Submit**: sezioni chiuse all'apertura.
   *Try it* mostra forme, loss e numero di parametri (come NNModelling), non il voto.
3. **Submit**: la riga `# study_checks.submit(...)` è commentata, così "Run
   all" non consegna. Si toglie il `#` e si esegue una volta: non stampa il
   risultato, conta solo il primo tentativo (gli altri sono registrati come
   `resubmit`) e scarica `nnm-study-<task>-<id>-<ora>-<tentativo>.json` nella
   cartella Download dell'esperto (codice eseguito, log, risultato).

I notebook si generano con `tools/expert_study/build_notebooks.py`; non
modificarli a mano. Dopo il merge in `main`: `build_notebooks.py --ref main` e
cambiare `expert-study-materials` in `main` nei link qui sopra e in
`prepare_usb.py`. I notebook nascondono Gemini
(`colab.generative_ai_disabled`); in più l'esperto disattiva i completamenti
AI in Tools → Settings → AI Assistance. Colab funziona solo con quello che è
pushato su GitHub.

## Prima del laboratorio

```sh
just build                                   # anche build/qt/study_probe
just test-expert-study
.venv/bin/python tools/expert_study/prepare_usb.py /run/media/$USER/Chiavetta/nnm-study
```

Sulla pennetta (`nnm-study/`): `master/` (copia per partecipante),
`facilitator/` (soluzioni, **da non mettere sul drive condiviso**),
`new_session.py`, `collect.sh`. Sul drive condiviso vanno solo `master/`,
`new_session.py` e `collect.sh`. Sulle macchine servono NNModelling installato
e un browser per Colab. L'ID di ogni partecipante è un
numero assegnato dal gruppo di ricerca (unico tra studenti ed esperti, es.
`7`): lo stesso per foglio dei task (riquadro in alto, a penna), cartella,
campo `participant_id` della prima cella Colab, file di consegna e log.
Nessun nome nei file: la tabella ID ↔ persona la tiene il gruppo di ricerca.

## Per ogni partecipante (`D` = cartella `nnm-study` sul drive)

1. `mv ~/nnmodelling.log ~/nnmodelling.log.old` (se esiste).
2. Scrivere l'ID (es. `7`) nel riquadro in alto del foglio dei task, poi
   `python3 D/new_session.py 7` → crea `~/nnm-expert-study/7/` con progetti,
   foglio e notebook, e stampa i comandi per quell'ID. Aprire sempre i
   progetti di questa cartella.
3. Ordine bilanciato: metà PyTorch→NNModelling, metà NNModelling→PyTorch.
   Ogni task NNModelling si apre dal proprio progetto `projects/taskN-…`; i
   task PyTorch dai link Colab in `notebooks/README.txt`.
4. Dopo ogni task PyTorch: spostare `~/Downloads/nnm-study-*.json` in
   `~/nnm-expert-study/7/notebooks/` (i file si chiamano `nnm-study-7-…`).
5. Alla fine: `bash D/collect.sh --reset ~/nnm-expert-study/7 7`
   (scrive `7.log`), poi copiare la cartella del partecipante sulla pennetta.

## Correzione e analisi

```sh
.venv/bin/python tools/expert_study/grade.py sessions/7 --csv grades.csv
```

Un task NNModelling è corretto se l'analisi nativa non ha problemi e il grafo,
eseguito in PyTorch dal runtime, supera gli stessi controlli dei notebook
(forme, loss, numero di parametri, nessun costrutto vietato, numero di teste,
fattore di scala 1/√(d/h); nel T2 con (d, h) = (32,4), (64,8) e (32,2)).
Una consegna Colab il cui ID non coincide con la cartella fallisce il
controllo `participant id`: va spostata nella cartella giusta.

Grafici di un partecipante:

```sh
python3 tools/expert_study/plot_session.py sessions/7/7.log --out analysis/7
```
