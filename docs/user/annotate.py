#!/usr/bin/env python3
"""Make cropped, numbered callout copies of the computer-use guide captures."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / ".computer-use" / "user-docs"
OUTPUT_DIR = ROOT / "docs" / "user" / "assets"
PAD = 56
RED = (205, 35, 35)
WHITE = (255, 255, 255)

# Crop rectangles use screenshot coordinates (left, top, right, bottom).
# Callout bubble and target points also use screenshot coordinates. Bubbles sit
# in the added white margin so the source controls remain unobstructed.
SHOTS: dict[str, dict] = {
    "overview": {
        "crop": (0, 50, 1360, 900),
        "caption": "Main workspace areas",
        "callouts": [
            (1, (130, 22), (130, 70), "Menu bar"),
            (2, (400, 22), (400, 116), "Scope, training and canvas toolbar"),
            (3, (-28, 235), (100, 235), "Package browser"),
            (4, (650, 928), (650, 810), "Graph canvas"),
            (5, (1388, 170), (1160, 170), "Inspector"),
            (6, (1388, 490), (1170, 540), "Project resources"),
            (7, (1388, 730), (1170, 770), "Model problems"),
            (8, (940, 928), (940, 880), "Status area"),
        ],
    },
    "inspector": {
        "crop": (0, 50, 1360, 900),
        "caption": "Inspector and project resources",
        "callouts": [
            (1, (1388, 220), (1100, 260), "Selected object properties"),
            (2, (1388, 550), (1120, 600), "Project resources and dataset actions"),
            (3, (1388, 745), (1120, 790), "Model problems"),
        ],
    },
    "menu-file": {
        "crop": (0, 50, 430, 315), "caption": "File menu",
        "callouts": [(1, (445, 92), (115, 150), "Create, open, save and close project")],
    },
    "menu-template": {
        "crop": (0, 50, 470, 250), "caption": "Project templates",
        "callouts": [(1, (485, 95), (342, 170), "Available templates")],
    },
    "menu-edit": {
        "crop": (0, 50, 400, 220), "caption": "Edit menu",
        "callouts": [(1, (415, 100), (120, 145), "Undo and redo")],
    },
    "menu-model": {
        "crop": (75, 50, 355, 300), "caption": "Model menu",
        "callouts": [(1, (370, 100), (195, 170), "Stereotype, dataset and training actions")],
    },
    "menu-view": {
        "crop": (130, 50, 430, 300), "caption": "View menu",
        "callouts": [(1, (445, 100), (260, 180), "Fit and zoom controls")],
    },
    "menu-arrange": {
        "crop": (130, 50, 510, 320), "caption": "Arrange submenu",
        "callouts": [(1, (525, 100), (425, 250), "Vertical and horizontal layouts")],
    },
    "scope-tree": {
        "crop": (0, 50, 350, 345), "caption": "Scope navigation",
        "callouts": [(1, (365, 115), (150, 240), "Root and nested scopes")],
    },
    "attention": {
        "crop": (0, 50, 1360, 900), "caption": "Attention subflow graph",
        "callouts": [
            (1, (280, 22), (300, 115), "Current scope"),
            (2, (1388, 500), (1140, 540), "Dataset used by the project"),
        ],
    },
    "dataset-manager": {
        "crop": (350, 205, 990, 665), "caption": "Dataset manager",
        "callouts": [
            (1, (330, 390), (640, 250), "Active dataset"),
            (2, (1010, 610), (520, 615), "Edit the selected dataset"),
        ],
    },
    "dataset-form": {
        "crop": (280, 115, 1060, 760), "caption": "Dataset metadata and input slots",
        "callouts": [
            (1, (265, 185), (620, 270), "Dataset identity and description"),
            (2, (1080, 430), (700, 490), "Input slot name, type and shape"),
            (3, (1080, 715), (905, 710), "Save changes or cancel"),
        ],
    },
    "dataset-targets": {
        "crop": (280, 115, 1060, 760), "caption": "Dataset target slots",
        "callouts": [
            (1, (265, 360), (650, 455), "Target slot name, type and shape"),
            (2, (1080, 715), (905, 710), "Save changes or cancel"),
        ],
    },
    "stereotype-form": {
        "source": "stereotype-form-full",
        "crop": (240, 35, 1085, 1260), "caption": "Modulo completo di definizione dello stereotipo",
        "callouts": [
            (1, (220, 165), (700, 180), "Identità, versione, nome, descrizione e tipo"),
            (2, (1115, 510), (690, 500), "Maniglie di uscita esplicite"),
            (3, (1115, 680), (700, 665), "Schema dei parametri"),
            (4, (220, 850), (680, 840), "Dipendenze del pacchetto"),
            (5, (1115, 1030), (700, 1025), "Funzione Lua opzionale"),
            (6, (650, 1290), (850, 1205), "Crea e salva il progetto"),
        ],
    },
    "project-id": {
        "crop": (540, 335, 800, 530), "caption": "Choose a project ID",
        "callouts": [
            (1, (525, 400), (660, 430), "Project ID"),
            (2, (815, 510), (655, 480), "Confirm"),
        ],
    },
    "project-name": {
        "crop": (540, 335, 800, 530), "caption": "Choose a display name",
        "callouts": [
            (1, (525, 400), (660, 430), "Display name"),
            (2, (815, 510), (655, 480), "Confirm"),
        ],
    },
    "training-connect": {
        "source": "training-dashboard-new",
        "crop": (0, 0, 1230, 95), "caption": "Connessione e configurazione del backend di addestramento",
        "callouts": [
            (1, (300, -25), (300, 10), "Backend endpoint"),
            (2, (800, -25), (800, 10), "Connect and check service health"),
            (3, (1245, 35), (1218, 35), "Open separate backend configuration"),
        ],
    },
    "training-settings": {
        "source": "training-dashboard-new",
        "crop": (100, 95, 1230, 155), "caption": "Impostazione delle opzioni di addestramento",
        "details": "10 epoche; batch 32; tasso di apprendimento 0,001; seme 0; pubblicazione ogni 10 passaggi.",
        "callouts": [
            (1, (128, 82), (128, 103), "Current training settings"),
            (2, (1250, 125), (1218, 125), "Save the project and submit the job"),
        ],
    },
    "slurm-configure": {
        "source": "slurm-configure",
        "crop": (330, 110, 1000, 650),
        "caption": "Configurazione dell’esecutore SSH/Slurm",
        "callouts": [
            (1, (300, 225), (390, 225), "Executor selection"),
            (2, (300, 300), (390, 300), "Slurm SSH alias"),
            (3, (300, 340), (390, 340), "Absolute remote job root"),
            (4, (300, 376), (390, 376), "Prebuilt Singularity image"),
            (5, (300, 414), (390, 414), "Partition and resource limits"),
            (6, (300, 565), (390, 565), "Runtime readiness status"),
            (7, (300, 610), (390, 610), "Safely restart the local backend"),
        ],
    },
    "slurm-dashboard": {
        "source": "training-dashboard-new",
        "crop": (0, 0, 1230, 95),
        "caption": "Dashboard di addestramento con backend Slurm connesso",
        "callouts": [
            (1, (300, -25), (300, 10), "Backend endpoint"),
            (2, (800, -25), (800, 10), "Connect and check service health"),
            (3, (1245, 35), (1218, 35), "Open separate backend configuration"),
        ],
    },
    "training-dashboard": {
        "crop": (110, 50, 1230, 900),
        "caption": "Dashboard al termine dell’addestramento in 12 passaggi",
        "details": "Completato in 3 epoche; perdita finale sul test 3,345268.",
        "callouts": [
            (1, (250, 22), (400, 130), "Backend connection"),
            (2, (600, 22), (650, 175), "Training configuration"),
            (3, (82, 350), (250, 310), "Job history and status"),
            (4, (1256, 380), (1150, 380), "Training and validation loss curves"),
            (5, (900, 22), (860, 230), "Curve toggles and scale"),
            (6, (1256, 185), (1120, 225), "Final test loss"),
            (7, (1256, 690), (650, 680), "Training and worker log"),
            (8, (930, 928), (1085, 865), "Download, restore, cancel and close actions"),
        ],
    },
    "training-scale": {
        "crop": (110, 50, 1230, 900), "caption": "Choose the learning-curve scale",
        "callouts": [
            (1, (1256, 250), (925, 270), "Linear or logarithmic curve scale"),
        ],
    },
    "download-wheel": {
        "crop": (110, 50, 1230, 900), "caption": "Save a downloaded wheel file",
        "callouts": [
            (1, (955, 560), (820, 585), "Wheel filename and destination"),
            (2, (990, 650), (885, 585), "Save the downloaded wheel"),
        ],
    },
    "subflow-menu": {
        "crop": (440, 390, 760, 565), "caption": "Subflow context menu",
        "callouts": [
            (1, (420, 485), (575, 480), "Enter the subflow"),
            (2, (780, 535), (615, 523), "Expand and review")
        ],
    },
    "decoder": {
        "crop": (0, 50, 1360, 900), "caption": "Decoder subflow graph",
        "callouts": [
            (1, (450, 22), (175, 115), "Current nested scope"),
            (2, (1388, 330), (570, 310), "Repeated subflow block"),
            (3, (1388, 690), (590, 650), "Projection and activation layers"),
        ],
    },
    "inspector-reference": {
        "crop": (0, 50, 1360, 900), "caption": "Inspector package reference",
        "callouts": [
            (1, (1388, 450), (1190, 485), "Referenced package"),
            (2, (1388, 535), (1190, 530), "Reference parameters"),
        ],
    },
    "inspector-reference-menu": {
        "crop": (850, 300, 1360, 620), "caption": "Choose a package reference",
        "callouts": [
            (1, (1388, 430), (1190, 485), "Available package choices"),
        ],
    },
    "inspector-boundary": {
        "crop": (0, 50, 1360, 900), "caption": "Map a subflow output boundary",
        "callouts": [
            (1, (1388, 405), (1190, 405), "Boundary mapping selector"),
            (2, (1388, 815), (500, 820), "Selected output boundary"),
        ],
    },
    "join-controls": {
        "crop": (380, 270, 620, 445), "caption": "Join input controls",
        "callouts": [
            (1, (362, 337), (458, 337), "Remove an input slot"),
            (2, (513, 250), (513, 337), "Add an input slot"),
        ],
    },
    "join-extra": {
        "crop": (380, 270, 620, 445), "caption": "Join input controls with connected inputs",
        "callouts": [
            (1, (362, 337), (458, 337), "Remove an input slot"),
            (2, (513, 250), (513, 337), "Add an input slot"),
        ],
    },
    "model-problems": {
        "crop": (0, 50, 1360, 900), "caption": "Incomplete model diagnostics",
        "callouts": [
            (1, (1388, 805), (1120, 840), "Problem list"),
            (2, (720, 928), (590, 645), "Affected graph nodes"),
        ],
    },
    "project-chooser": {
        "crop": (340, 350, 1000, 520), "caption": "Choose a project to edit",
        "callouts": [
            (1, (320, 445), (430, 458), "Create a new project"),
            (2, (1020, 445), (840, 458), "Open an existing project"),
        ],
    },
    "stereotype-form-top": {
        "source": "stereotype-form-full",
        "crop": (240, 35, 1085, 650), "caption": "Identità e uscite dello stereotipo",
        "callouts": [
            (1, (220, 165), (700, 175), "Identificativo, versione, nome e descrizione"),
            (2, (1115, 430), (690, 500), "Tipo, colore e uscite"),
        ],
    },
    "stereotype-form-bottom": {
        "source": "stereotype-form-full",
        "crop": (240, 575, 1085, 1260), "caption": "Parametri, dipendenze e funzione Lua",
        "callouts": [
            (1, (220, 665), (700, 665), "Schema dei parametri"),
            (2, (1115, 845), (680, 835), "Dipendenze del pacchetto"),
            (3, (1115, 1020), (700, 1025), "Funzione Lua opzionale"),
            (4, (650, 1290), (850, 1205), "Crea e salva il progetto"),
        ],
    },
}

ITALIAN_LABELS = {
    "Active dataset": "Dataset attivo",
    "Add an input slot": "Aggiungi uno slot di ingresso",
    "Affected graph nodes": "Nodi del grafo interessati",
    "Arrange submenu": "Sottomenu Disponi",
    "Attention subflow graph": "Grafo del sottoflusso di attenzione",
    "Available package choices": "Pacchetti disponibili",
    "Available templates": "Modelli disponibili",
    "Backend connection": "Connessione al backend",
    "Backend endpoint": "Indirizzo del backend",
    "Current training settings": "Impostazioni correnti dell’addestramento",
    "Executor selection": "Selezione dell’esecutore",
    "Absolute remote job root": "Directory remota assoluta delle attività",
    "Prebuilt Singularity image": "Immagine Singularity precompilata",
    "Partition and resource limits": "Partizione e limiti delle risorse",
    "Runtime readiness status": "Stato di disponibilità del runtime",
    "Safely restart the local backend": "Riavvia in sicurezza il backend locale",
    "Open separate backend configuration": "Apri la configurazione separata del backend",
    "Slurm SSH alias": "Alias SSH di Slurm",
    "Slurm runtime readiness": "Stato di disponibilità del runtime Slurm",
    "10 epochs; batch size 32; learning rate 0.001; seed 0; publish every 10 steps.":
        "10 epoche; batch 32; tasso di apprendimento 0,001; seme 0; pubblicazione ogni 10 passaggi.",
    "Boundary mapping selector": "Selettore della mappatura del confine",
    "Choose a display name": "Scegli un nome visualizzato",
    "Choose a package reference": "Scegli un riferimento a un pacchetto",
    "Choose a project ID": "Scegli un identificativo per il progetto",
    "Choose a project to edit": "Scegli un progetto da modificare",
    "Choose the learning-curve scale": "Scegli la scala delle curve di apprendimento",
    "Confirm": "Conferma",
    "Connect and check service health": "Connetti e verifica lo stato del servizio",
    "Create a new project": "Crea un nuovo progetto",
    "Create, open, save and close project": "Crea, apri, salva e chiudi il progetto",
    "Current nested scope": "Ambito annidato corrente",
    "Current scope": "Ambito corrente",
    "Curve toggles and scale": "Visibilità e scala delle curve",
    "Dataset identity and description": "Identificativo, versione, nome e descrizione del dataset",
    "Dataset manager": "Gestione dei dataset",
    "Dataset metadata and input slots": "Metadati del dataset e slot di ingresso",
    "Dataset target slots": "Slot target del dataset",
    "Dataset used by the project": "Dataset usato dal progetto",
    "Decoder subflow graph": "Grafo del sottoflusso decoder",
    "Display name": "Nome visualizzato",
    "Download, restore, cancel and close actions": "Azioni per scaricare, ripristinare, annullare e chiudere",
    "Edit menu": "Menu Modifica",
    "Edit the selected dataset": "Modifica il dataset selezionato",
    "Enter the subflow": "Entra nel sottoflusso",
    "Epochs, batch size, learning rate and seed": "Epoche, dimensione del batch, tasso di apprendimento e seme",
    "Expand and review": "Espandi e ispeziona",
    "File menu": "Menu File",
    "Final test loss": "Perdita finale sul test",
    "Fit and zoom controls": "Comandi Adatta e zoom",
    "Graph canvas": "Canvas del grafo",
    "Incomplete model diagnostics": "Diagnostica del modello incompleto",
    "Input slot name, type and shape": "Nome, tipo e forma dello slot di ingresso",
    "Inspector": "Ispettore",
    "Inspector and project resources": "Ispettore e risorse del progetto",
    "Inspector package reference": "Riferimento a un pacchetto nel pannello Proprietà",
    "Job history and status": "Cronologia e stato delle attività",
    "Join input controls": "Comandi degli ingressi del nodo join",
    "Join input controls with connected inputs": "Comandi del nodo join con ingressi collegati",
    "Linear or logarithmic curve scale": "Scala lineare o logaritmica delle curve",
    "Main workspace areas": "Aree principali dell’area di lavoro",
    "Map a subflow output boundary": "Mappa il confine di uscita del sottoflusso",
    "Menu bar": "Barra dei menu",
    "Model menu": "Menu Modello",
    "Model problems": "Problemi del modello",
    "Open an existing project": "Apri un progetto esistente",
    "Package browser": "Catalogo dei pacchetti",
    "Problem list": "Elenco dei problemi",
    "Project ID": "Identificativo del progetto",
    "Project resources": "Risorse del progetto",
    "Project resources and dataset actions": "Risorse del progetto e azioni sui dataset",
    "Project templates": "Modelli di progetto",
    "Projection and activation layers": "Livelli di proiezione e attivazione",
    "Reference parameters": "Parametri del riferimento",
    "Referenced package": "Pacchetto referenziato",
    "Remove an input slot": "Rimuovi uno slot di ingresso",
    "Repeated subflow block": "Blocco sottoflusso ripetuto",
    "Root and nested scopes": "Ambito principale e ambiti annidati",
    "Save a downloaded wheel file": "Salva un file wheel scaricato",
    "Save changes or cancel": "Salva le modifiche o annulla",
    "Save the downloaded wheel": "Salva il wheel scaricato",
    "Save the project and submit the job": "Salva il progetto e invia l’attività",
    "Scope navigation": "Navigazione degli ambiti",
    "Scope, training and canvas toolbar": "Ambito, addestramento e strumenti del canvas",
    "Selected object properties": "Proprietà dell’oggetto selezionato",
    "Selected output boundary": "Confine di uscita selezionato",
    "Status area": "Area di stato",
    "Stereotype, dataset and training actions": "Azioni per stereotipi, dataset e addestramento",
    "Subflow context menu": "Menu contestuale del sottoflusso",
    "Target slot name, type and shape": "Nome, tipo e forma dello slot target",
    "Training and validation loss curves": "Curve della perdita di addestramento e validazione",
    "Training and worker log": "Registro dell’addestramento e del worker",
    "Training configuration": "Configurazione dell’addestramento",
    "Undo and redo": "Annulla e ripeti",
    "Vertical and horizontal layouts": "Disposizione verticale e orizzontale",
    "View menu": "Menu Visualizza",
    "Wheel filename and destination": "Nome e destinazione del file wheel",
    "3 epochs; batch size 16; learning rate 0.001; seed 0; publish every step.":
        "3 epoche; batch 16; tasso di apprendimento 0,001; seme 0; pubblicazione a ogni passaggio.",
    "Completed in 3 epochs; final test loss 3.345268.":
        "Completato in 3 epoche; perdita finale sul test 3,345268.",
}


def italian(text: str) -> str:
    return ITALIAN_LABELS.get(text, text)


def font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start, end), fill=RED, width=4)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    base = (end[0] - 13 * ux, end[1] - 13 * uy)
    draw.polygon([
        tip,
        (base[0] + 7 * px, base[1] + 7 * py),
        (base[0] - 7 * px, base[1] - 7 * py),
    ], fill=RED)


def make_one(stem: str, spec: dict) -> list[dict]:
    source = SOURCE_DIR / f"{spec.get('source', stem)}.jpg"
    image = Image.open(source).convert("RGB")
    left, top, right, bottom = spec["crop"]
    crop = image.crop((left, top, right, bottom))
    canvas = Image.new("RGB", (crop.width + 2 * PAD, crop.height + 2 * PAD), WHITE)
    canvas.paste(crop, (PAD, PAD))
    draw = ImageDraw.Draw(canvas)
    entries = []
    for number, bubble, target, description in spec["callouts"]:
        bx, by = bubble[0] - left + PAD, bubble[1] - top + PAD
        tx, ty = target[0] - left + PAD, target[1] - top + PAD
        radius = 17
        dx, dy = tx - bx, ty - by
        length = max(1, (dx * dx + dy * dy) ** 0.5)
        start = (round(bx + dx / length * radius), round(by + dy / length * radius))
        add_arrow(draw, start, (tx, ty))
        draw.ellipse((bx - radius, by - radius, bx + radius, by + radius),
                     fill=WHITE, outline=RED, width=4)
        label = str(number)
        box = draw.textbbox((0, 0), label, font=font(19))
        draw.text((bx - (box[2] - box[0]) / 2, by - (box[3] - box[1]) / 2 - box[1]),
                  label, font=font(19), fill=RED)
        entries.append({"number": number, "region": description})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_DIR / f"{stem}.png", optimize=True)
    return entries


def main() -> None:
    captions = {}
    for stem, spec in SHOTS.items():
        captions[f"{stem}.png"] = {
            "caption": italian(spec["caption"]),
            "callouts": [
                {"number": item["number"], "region": italian(item["region"])}
                for item in make_one(stem, spec)
            ],
        }
        if "details" in spec:
            captions[f"{stem}.png"]["details"] = italian(spec["details"])
    (OUTPUT_DIR / "captions.json").write_text(
        json.dumps(captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
