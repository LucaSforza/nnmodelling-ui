# Studio esperti — materiali (proposta)

Specifica e criteri: `docs/knowledge/testing/expert-study.md`.
Report riassuntivo con screenshot: `report/expert-study-report.pdf`.

| Cartella | Contenuto |
| --- | --- |
| `tasks/expert-tasks.tex` | Foglio unico dei task (inglese) per NNModelling e PyTorch |
| `projects/<task>/start` | Progetto NNModelling da cui parte l'esperto |
| `projects/<task>/solution` | Soluzione di riferimento (solo per il facilitatore) |
| `notebooks/` | Notebook PyTorch dei partecipanti; `solutions/` per il facilitatore |
| `report/` | Report LaTeX e screenshot |

Strumenti in `tools/expert_study/`: `study_checks.py` (controlli PyTorch,
copiato accanto ai notebook), `grade.py` (correzione automatica di progetti e
notebook), `prepare_usb.py` (pennetta), `split_activity_log.py` e
`analyse_activity.py` (tempi per task dal log di NNModelling).

## Prima del laboratorio

```sh
just build                                   # anche build/qt/study_probe
just test-expert-study                       # 56+ test
.venv/bin/python tools/expert_study/prepare_usb.py /media/USB/nnm-study
```

Sulla pennetta: `master/` (copia per partecipante), `facilitator/`
(soluzioni, da non mostrare), `new_session.py`. Sulle macchine serve un kernel
Jupyter con `torch` (nient'altro) e NNModelling installato.

## Per ogni partecipante

1. `mv ~/nnmodelling.log ~/nnmodelling.log.old` (se esiste).
2. `python3 /media/USB/nnm-study/new_session.py P07` → crea
   `~/nnm-expert-study/P07/` con progetti, notebook e foglio.
3. Ordine bilanciato: metà PyTorch→NNModelling, metà NNModelling→PyTorch.
   Ogni task NNModelling si apre dal proprio progetto `projects/taskN-…`.
4. Alla fine: copiare `~/nnm-expert-study/P07/` e
   `bash tools/activity_log/collect.sh logs/` (log anonimo).

## Correzione e analisi

```sh
.venv/bin/python tools/expert_study/grade.py sessions/P07 --csv grades.csv
```

Un task NNModelling è corretto se l'analisi nativa non ha problemi e il grafo,
eseguito in PyTorch dal runtime, supera gli stessi controlli dei notebook
(forme, loss, numero di parametri, nessun costrutto vietato, numero di teste;
per il T3 anche con `d-model`/`num-heads` diversi).
