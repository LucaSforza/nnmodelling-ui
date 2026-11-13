# Raccolta e analisi del log di attivita

Il client scrive in append `~/nnmodelling.log` (o il file indicato da
`NNMODELLING_ACTIVITY_LOG`). Contratto: `docs/knowledge/contracts/activity-log.md`.

## Flusso in laboratorio (Ubuntu)

1. Prima dello studio, su ogni macchina azzera il log:
   `mv ~/nnmodelling.log ~/nnmodelling.log.old` (se esiste).
2. Gli studenti eseguono i dieci task partendo da `examples/test`.
3. Raccolta, in locale: `bash tools/activity_log/collect.sh logs/`
   oppure da remoto: `ssh host 'bash -s -- /tmp/logs' < tools/activity_log/collect.sh`
   e poi `scp host:/tmp/logs/*.jsonl logs/`. Con `--reset` il log originale viene
   spostato in `~/nnmodelling.log.collected-<timestamp>`.
   Il file copiato ha un nome casuale `P-<hex>.jsonl`: e l'unico identificativo
   del partecipante (anonimo).
4. Analisi:
   `python3 tools/activity_log/parse_activity_log.py logs/*.jsonl --out results/`
   produce `events.csv`, `sessions.csv`, `tasks.csv`, `summary.md`.
   Opzioni: `--tasks FILE` (default `tasks_bachelor.json`), `--idle-seconds 30`.

Le righe malformate o troncate e i salti di `seq` vengono saltati e riportati
su stderr e in `summary.md`.

## Report per il paper e prove a vuoto

- `python3 tools/activity_log/paper_report.py results/ report/` legge i CSV del
  parser e scrive `report/tables/*.tex`, `report/figures/*.pdf`, `measures.csv`
  e (con `--body FILE.tex`) `report.pdf`. Richiede matplotlib e pdflatex.
- `python3 tools/activity_log/simulate_sessions.py logs/` genera 5 log SINTETICI
  (archetipi: fluente, vocabolario, collegamenti, parametri, interrotto) per
  provare la pipeline; usare `--synthetic` in `paper_report.py` per marcare il PDF.
