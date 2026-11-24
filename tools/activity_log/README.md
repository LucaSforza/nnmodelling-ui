# Raccolta e analisi del log di attivita

Il client scrive in append `~/nnmodelling.log` (o il file indicato da
`NNMODELLING_ACTIVITY_LOG`). Contratto: `docs/knowledge/contracts/activity-log.md`.

## Flusso in laboratorio (Ubuntu)

1. Prima dello studio, su ogni macchina azzera il log:
   `mv ~/nnmodelling.log ~/nnmodelling.log.old` (se esiste).
2. Ogni partecipante ha un ID numerico assegnato dal gruppo di ricerca (unico
   tra studenti ed esperti), scritto dal facilitatore nel riquadro in alto del
   foglio dei task (`study/students/tasks/student-tasks.pdf`). Gli studenti
   eseguono i task partendo da `examples/test`.
3. Raccolta in laboratorio con la chiavetta: `python3 <chiavetta>/nnm-study/collect.py ID student`
   (vedi il `README.md` della chiavetta e `study/experts/README.md`).
   Solo log, in locale: `bash tools/activity_log/collect.sh logs/` (chiede l'ID)
   oppure `bash tools/activity_log/collect.sh logs/ 7`;
   da remoto: `ssh host 'bash -s -- /tmp/logs 7' < tools/activity_log/collect.sh`
   e poi `scp host:/tmp/logs/*.log logs/`. Con `--reset` il log originale viene
   spostato in `~/nnmodelling.log.collected-<timestamp>`.
   Il file copiato si chiama `<id>.log` (solo cifre, senza zeri iniziali:
   `007` → `7.log`): è l'identificativo del partecipante nell'analisi. Non
   sovrascrive mai un file esistente. Nessun nome nei file: la tabella ID ↔
   persona la tiene il gruppo di ricerca.
4. Analisi:
   `python3 tools/activity_log/parse_activity_log.py logs/*.log --out results/`
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
