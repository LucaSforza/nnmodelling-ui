#!/usr/bin/env python3
"""Prepare the expert-study USB stick (docs/knowledge/testing/expert-study.md).

Usage: prepare_usb.py [--force] OUT

Writes OUT/master (start projects, start notebooks and README.txt with the Colab links,
task sheet), OUT/README.md, OUT/new_session.py ID (copies master/ into a fresh
participant folder <id>/), OUT/collect.py, OUT/collect.sh and OUT/data/. ID is the
numeric participant ID. No solutions: they stay in the repository only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT / "study" / "experts"
TASKS = ("task1-mlp", "task2-multihead")
sys.path.insert(0, str(HERE))
from build_notebooks import DEFAULT_REF, REPO, TASKS as NOTEBOOKS  # noqa: E402

COLAB = f"https://colab.research.google.com/github/{REPO}/blob/{DEFAULT_REF}/study/experts/notebooks/"
NOTEBOOKS_README = """PyTorch tasks run in Google Colab (browser, the expert's Google account):

""" + "".join(f"  {task}: {COLAB}{spec['file']}\n" for task, spec in NOTEBOOKS.items()) + """
The .ipynb files here are the same start notebooks, as a fallback: in Colab use
File > Upload notebook (the first cell still needs internet to reach GitHub).

Submitting a task downloads nnm-study-<task>-<participant>-<time>.json into the
expert's Downloads folder. Move that file into this folder, then run grade.py.
"""
USB_README = """# Studio NNModelling — cosa fare ogni volta che colleghi la chiavetta

Tutti i comandi si lanciano da un terminale. La chiavetta non conserva i
permessi: usa sempre `python3 …` o `bash …`, mai `./…`.

Per prima cosa, in ogni terminale, trova la cartella `nnm-study` della chiavetta:

```sh
D="$(ls -d /media/$USER/*/nnm-study /run/media/$USER/*/nnm-study 2>/dev/null | head -1)"; echo "$D"
```

Se non stampa niente, apri la chiavetta nel file manager e scrivi a mano
`D=/percorso/della/chiavetta/nnm-study`.

Ogni partecipante ha un **ID numerico** (solo cifre, unico tra studenti ed
esperti). Scrivilo nel riquadro in alto del foglio dei task. Nei file non va
mai il nome: la tabella ID ↔ persona la tieni tu, a parte.

## 1. PC del laboratorio — prima di ogni partecipante

```sh
[ -f ~/nnmodelling.log ] && mv ~/nnmodelling.log ~/nnmodelling.log.old-$(date +%H%M%S)
```

- **Studente**: apri NNModelling sul progetto dei task come al solito.
- **Esperto** (esempio con ID 7):

  ```sh
  python3 "$D/new_session.py" 7
  ```

  Crea `~/nnm-expert-study/7/`: i progetti NNModelling si aprono **solo da lì**.
  I task PyTorch si fanno su Colab, nel browser:
""" + "".join(f"  - {task}: {COLAB}{spec['file']}\n" for task, spec in NOTEBOOKS.items()) + """
  Al primo ▶ Colab chiede conferma: **Run anyway**. Nella prima cella va l'ID.
  Ordine bilanciato: metà PyTorch→NNModelling, metà NNModelling→PyTorch.

## 2. PC del laboratorio — quando il partecipante ha finito

Prima **chiudi NNModelling** (e, per gli esperti, controlla con Ctrl+J in
Chrome che le due consegne Colab siano state scaricate). Poi:

```sh
python3 "$D/collect.py" 7 student     # oppure: python3 "$D/collect.py" 7 expert
```

Copia sulla chiavetta, in `$D/data/<student|expert>/7/`:
- il log di NNModelling (`7.log`);
- per gli esperti, anche i progetti (`projects/`) e le consegne Colab
  (`colab/nnm-study-7-T1-….json`, `…-T2-….json`, cercate in `~/Scaricati` e `~/Downloads`).

Poi sposta da parte log e consegne, così il partecipante successivo sullo
stesso PC parte pulito. **Leggi l'output**: `ok` per ogni cosa trovata,
`MISSING` per quello che manca. Deve finire con `DONE`.

- `MISSING Colab submission T…`: in Chrome premi Ctrl+J e guarda dove è finito il
  file `nnm-study-7-T…-….json`; copialo a mano in `$D/data/expert/7/colab/`
  (crea la cartella se manca).
- `already collected`: quell'ID è già sulla chiavetta. Niente viene sovrascritto:
  controlla di aver scritto l'ID giusto.
- `nothing found`: nessun dato per quell'ID su questo PC (ID sbagliato, o PC sbagliato).

## 3. Il tuo computer — a fine giornata

Dalla cartella del repository `nnmodelling-ui`, con la chiavetta collegata:

```sh
.venv/bin/python tools/expert_study/import_usb.py "$D/data" ~/nnm-study-data \\
    --participants ~/nnm-study-data/participants.csv --grade
```

`participants.csv` ha le colonne `id,group` (`group` = `student` o `expert`),
una riga per partecipante; senza `--participants` il controllo degli ID salta.
Risultato:
- `~/nnm-study-data/student/<id>/<id>.log`
- `~/nnm-study-data/expert/<id>/{<id>.log, projects/, colab/}`
- `~/nnm-study-data/grades.csv` (esperti) e `~/nnm-study-data/analysis/students/` (studenti).

Puoi rilanciarlo quando vuoi: i file già importati vengono saltati, niente
viene sovrascritto. Ogni `PROBLEM:` va controllato a mano.

## 4. Il tuo computer — grafici e un unico PDF con tutti i risultati

Sempre dalla cartella del repository, dopo il passo 3:

```sh
uv run --with matplotlib python tools/expert_study/report_all.py ~/nnm-study-data
xdg-open ~/nnm-study-data/report/results.pdf
```

`results.pdf` contiene: tabella dei partecipanti (cosa è stato raccolto per
ogni ID), voti degli esperti per tool e task con i controlli falliti e i
tempi, le figure dell'attività NNModelling di ogni esperto e il report degli
studenti. Figure e tabelle singole restano in `~/nnm-study-data/report/`
(`experts/<id>/`, `students/figures/`, `students/tables/`).
Rilancialo dopo ogni nuovo import: rigenera il PDF (i voti si ricalcolano se
cancelli `~/nnm-study-data/grades.csv`).

## Prova con i dati simulati (ID 9xx)

Se sulla chiavetta ci sono partecipanti `9xx` (studenti 901, 903, 905, 907,
909, 910; esperti 902, 904, 906, 908), sono **simulati** con
`tools/expert_study/simulate_lab.py`. Servono a provare i passi 3 e 4, non
sono dati dello studio:

```sh
.venv/bin/python tools/expert_study/import_usb.py "$D/data" ~/nnm-study-sim-data \\
    --participants "$D/participants-sim.csv" --grade
uv run --with matplotlib python tools/expert_study/report_all.py ~/nnm-study-sim-data
xdg-open ~/nnm-study-sim-data/report/results.pdf
```

**Prima del laboratorio cancellali** dalla chiavetta:

```sh
rm -r "$D"/data/student/9?? "$D"/data/expert/9?? "$D/participants-sim.csv"
```

(Se te ne dimentichi, `import_usb.py` con il `participants.csv` vero li
segnala come `not in the participants list`.)

## Regole

- Sulla chiavetta non ci sono soluzioni: restano solo nel repository.
- Non cancellare `data/` dalla chiavetta finché non hai importato e fatto una copia.
"""

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")

NEW_SESSION = '''#!/usr/bin/env python3
"""Create a fresh participant folder from master/ (standard library only).

Usage: new_session.py [--dest DIR] ID   (ID = numeric participant ID, digits only)
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

TASKS = ("task1-mlp", "task2-multihead")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="~/nnm-expert-study", help="folder holding one directory per participant")
    parser.add_argument("participant", metavar="ID")
    args = parser.parse_args()
    text = args.participant.strip()
    if not re.fullmatch(r"[0-9]{1,9}", text) or int(text) == 0:
        sys.exit("ID must be the numeric participant ID (digits only, not 0)")
    pid = str(int(text))  # canonical: 007 -> 7
    here = Path(__file__).resolve().parent
    master = here / "master"
    if not master.is_dir():
        sys.exit(f"missing {master}")
    target = Path(args.dest).expanduser() / pid
    if target.exists():
        sys.exit(f"refusing to overwrite existing participant folder: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(master, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Participant ID: {pid} (write it on the task sheet; the expert types it in Colab)")
    print(f"Participant folder: {target}")
    for task in TASKS:
        print(f"NNModelling project: {target / 'projects' / task}")
    print(f"Colab links: {target / 'notebooks' / 'README.txt'}; submissions stay in the browser's "
          "download folder (~/Scaricati or ~/Downloads), collect.py finds them")
    print(f"At the end: python3 {here / 'collect.py'} {pid} expert")


if __name__ == "__main__":
    main()
'''


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"missing study material: {source}")
    shutil.copytree(source, target, ignore=IGNORE)


def build_task_sheet(master: Path) -> None:
    tex = STUDY / "tasks" / "expert-tasks.tex"
    target = master / "tasks"
    target.mkdir(parents=True, exist_ok=True)
    if not tex.is_file():
        print(f"warning: {tex} not found, no task sheet", file=sys.stderr)
        return
    if shutil.which("pdflatex"):
        with tempfile.TemporaryDirectory(prefix="nnm-tasks-") as temporary:
            work = Path(temporary)
            shutil.copy(tex, work / tex.name)
            for sibling in tex.parent.iterdir():
                if sibling.is_dir():  # e.g. img/ with the screenshots
                    shutil.copytree(sibling, work / sibling.name)
                elif sibling != tex and sibling.suffix != ".pdf":
                    shutil.copy(sibling, work / sibling.name)
            ok = True
            for _ in range(2):  # second pass resolves references
                done = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                    cwd=work, capture_output=True, text=True, timeout=180,
                )
                ok = ok and done.returncode == 0
            pdf = work / "expert-tasks.pdf"
            if ok and pdf.is_file():
                shutil.copy(pdf, target / "expert-tasks.pdf")
                return
        print("warning: pdflatex failed, copying the .tex instead", file=sys.stderr)
    else:
        print("warning: pdflatex not found, copying the .tex instead", file=sys.stderr)
    shutil.copy(tex, target / tex.name)


def prepare(out: Path, force: bool = False) -> None:
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} exists and is not a directory")
    if out.is_dir() and any(out.iterdir()) and not force:
        raise SystemExit(f"{out} is not empty (use --force to rebuild master/)")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("master", "facilitator"):  # facilitator/: solutions of older layouts, removed
        shutil.rmtree(out / name, ignore_errors=True)
    master = out / "master"
    for task in TASKS:
        copy_tree(STUDY / "projects" / task / "start", master / "projects" / task)
    notebooks = master / "notebooks"
    notebooks.mkdir(parents=True)
    (notebooks / "README.txt").write_text(NOTEBOOKS_README, encoding="utf-8")
    for spec in NOTEBOOKS.values():  # start notebooks only, never the solutions
        shutil.copy(STUDY / "notebooks" / spec["file"], notebooks / spec["file"])
    build_task_sheet(master)
    shutil.copy(ROOT / "tools" / "activity_log" / "collect.sh", out / "collect.sh")
    shutil.copy(HERE / "collect_participant.py", out / "collect.py")
    (out / "data").mkdir(exist_ok=True)  # collected data: never removed, even with --force
    (out / "README.md").write_text(USB_README, encoding="utf-8")
    script = out / "new_session.py"
    script.write_text(NEW_SESSION, encoding="utf-8")
    script.chmod(0o755)
    print(f"Prepared {out}: README.md, master/ (copy per participant), "
          "new_session.py, collect.py, data/")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="allow a non-empty OUT (rebuilds master/)")
    parser.add_argument("out", metavar="OUT")
    args = parser.parse_args(argv)
    prepare(Path(args.out).expanduser(), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
