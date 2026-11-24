"""Lab data collection onto the USB stick and import at home (docs/knowledge/testing/expert-study.md)."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "expert_study"
PROJECTS = ROOT / "study" / "experts" / "projects"
sys.path.insert(0, str(TOOLS))

import collect_participant as cp  # noqa: E402
import import_usb  # noqa: E402

TASKS = ("task1-mlp", "task2-multihead")


def machine(tmp_path, name="pc-1"):
    home = tmp_path / name
    home.mkdir()
    return home


def write_log(home, text='{"v":1,"event":"session.start"}\n'):
    (home / "nnmodelling.log").write_text(text)


def expert_session(home, pid, downloads="Scaricati", tasks=("T1", "T2"), other_ids=()):
    session = home / "nnm-expert-study" / pid
    for task in TASKS:
        shutil.copytree(PROJECTS / task / "solution", session / "projects" / task)
    (session / "notebooks").mkdir(parents=True)
    folder = home / downloads
    folder.mkdir(exist_ok=True)
    for task in tasks:
        (folder / f"nnm-study-{pid}-{task}-20261125T100000Z-1.json").write_text(json.dumps({"task": task}))
    for other in other_ids:
        (folder / f"nnm-study-{other}-T1-20261125T090000Z-1.json").write_text(json.dumps({"task": "T1"}))


def collect(home, data, *args):
    return subprocess.run([sys.executable, str(TOOLS / "collect_participant.py"), *args,
                           "--home", str(home), "--data", str(data)], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin"})


def test_two_students_on_the_same_machine_are_kept_apart(tmp_path):
    home, data = machine(tmp_path), tmp_path / "usb/data"
    write_log(home, "first\n")
    done = collect(home, data, "001", "student")
    assert done.returncode == 0 and "DONE" in done.stdout, done.stdout + done.stderr
    assert (data / "student/1/1.log").read_text() == "first\n"
    assert not (home / "nnmodelling.log").exists()  # moved aside for the next participant
    assert list(home.glob("nnmodelling.log.collected-*"))
    write_log(home, "second\n")
    assert collect(home, data, "3", "student").returncode == 0
    assert (data / "student/3/3.log").read_text() == "second\n"


def test_expert_collects_log_projects_and_colab_from_any_download_folder(tmp_path):
    for downloads in ("Scaricati", "Downloads"):
        home, data = machine(tmp_path, downloads), tmp_path / downloads / "usb"
        write_log(home)
        expert_session(home, "4", downloads, other_ids=("9",))
        done = collect(home, data, "4", "expert")
        assert done.returncode == 0, done.stdout + done.stderr
        target = data / "expert/4"
        assert (target / "4.log").is_file()
        assert all((target / "projects" / task / "model.json").is_file() for task in TASKS)
        assert sorted(p.name.split("-")[3] for p in (target / "colab").iterdir()) == ["T1", "T2"]
        assert "belongs to another ID" in done.stdout and not list(target.glob("colab/nnm-study-9-*"))
        assert not list((home / downloads).glob("nnm-study-4-*"))  # moved to nnm-study-collected/
        assert len(list((home / downloads / "nnm-study-collected").glob("nnm-study-4-*"))) == 2


def test_xdg_download_folder_is_searched(tmp_path):
    home, data = machine(tmp_path), tmp_path / "usb/data"
    write_log(home)
    expert_session(home, "6", "Telechargements")
    (home / ".config").mkdir()
    (home / ".config/user-dirs.dirs").write_text('XDG_DOWNLOAD_DIR="$HOME/Telechargements"\n')
    assert collect(home, data, "6", "expert").returncode == 0
    assert len(list((data / "expert/6/colab").iterdir())) == 2


def test_missing_items_are_reported_but_what_exists_is_collected(tmp_path):
    home, data = machine(tmp_path), tmp_path / "usb/data"
    write_log(home)
    expert_session(home, "2", tasks=("T1",))
    shutil.rmtree(home / "nnm-expert-study/2/projects/task2-multihead")
    done = collect(home, data, "2", "expert", "--keep")
    assert done.returncode == 1
    assert "MISSING  Colab submission T2" in done.stdout and "MISSING  NNModelling project T2" in done.stdout
    assert (data / "expert/2/4.log").exists() is False and (data / "expert/2/2.log").is_file()
    assert (home / "nnmodelling.log").is_file()  # --keep leaves everything in place


def test_never_overwrites_and_rejects_bad_ids(tmp_path):
    home, data = machine(tmp_path), tmp_path / "usb/data"
    write_log(home, "x\n")
    assert collect(home, data, "5", "student", "--keep").returncode == 0
    again = collect(home, data, "05", "student")
    assert again.returncode != 0 and "already collected" in again.stderr
    other = collect(home, data, "5", "expert")
    assert other.returncode != 0 and "already collected as student" in other.stderr
    for bad in ("P05", "0", "mario"):
        assert collect(home, data, bad, "student").returncode == 2
    empty = machine(tmp_path, "pc-2")
    nothing = collect(empty, data, "8", "student")
    assert nothing.returncode == 2 and "nothing found" in nothing.stdout and not (data / "student/8").exists()


def make_usb_data(tmp_path):
    data = tmp_path / "usb/data"
    for group, pid in (("student", "1"), ("expert", "2"), ("student", "3")):
        (data / group / pid).mkdir(parents=True)
        (data / group / pid / f"{pid}.log").write_text(f"log {pid}\n")
    return data


def write_participants(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "group"])
        writer.writerows(rows)
    return path


def test_import_copies_skips_identical_and_never_overwrites(tmp_path, capsys):
    data, dest = make_usb_data(tmp_path), tmp_path / "home/dati"
    assert import_usb.main([str(data), str(dest)]) == 0
    assert (dest / "student/1/1.log").read_text() == "log 1\n" and (dest / "expert/2/2.log").is_file()
    assert import_usb.main([str(data), str(dest)]) == 0  # rerun: identical files skipped
    assert "0 new file(s), 1 already there" in capsys.readouterr().out
    (data / "student/1/1.log").write_text("changed\n")
    assert import_usb.main([str(data), str(dest)]) == 1
    assert "nothing overwritten" in capsys.readouterr().out
    assert (dest / "student/1/1.log").read_text() == "log 1\n"


def test_import_checks_ids_against_the_participants_list(tmp_path, capsys):
    data, dest = make_usb_data(tmp_path), tmp_path / "dati"
    (data / "expert/3").mkdir()  # ID 3 collected in both groups
    (data / "expert/3/3.log").write_text("x\n")
    (data / "student/P9").mkdir()
    listed = write_participants(tmp_path / "participants.csv",
                                [("1", "student"), ("2", "student"), ("3", "student"), ("4", "expert")])
    assert import_usb.main([str(data), str(dest), "--participants", str(listed)]) == 1
    out = capsys.readouterr().out
    assert "ID 3 is both student and expert" in out
    assert "student/P9: not a numeric participant ID" in out
    assert "ID 4 (expert) is in the participants list but has no data" in out
    assert "ID 2 is student in the participants list but was collected as expert" in out


def test_report_all_builds_one_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    if not shutil.which("pdflatex"):
        pytest.skip("pdflatex missing")
    dest = tmp_path / "dati"
    (dest / "expert/2").mkdir(parents=True)
    (dest / "grades.csv").write_text("participant,tool,task,target,passed,failed_checks,detail,info\n"
                                     "2,pytorch,T1,x,True,,,10.0 min from start to submission\n"
                                     "2,nnm,T1,x,False,parameters,parameters: 100 trainable_,\n")
    write_participants(dest / "participants.csv", [("2", "expert"), ("5", "student")])
    done = subprocess.run([sys.executable, str(TOOLS / "report_all.py"), str(dest)], capture_output=True, text=True)
    assert (dest / "report/results.pdf").is_file(), done.stderr[-2000:]
    tex = (dest / "report/results.tex").read_text()
    assert "Listed in participants.csv but not collected: 5" in tex and r"trainable\_" in tex
