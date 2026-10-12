#!/usr/bin/env python3
"""Make cropped, numbered callout copies of the computer-use guide captures."""

from __future__ import annotations

import json
import argparse
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
        "source": "menu-model-current",
        "crop": (89, 0, 300, 293),
        "caption": "Model menu",
        "caption_en": "Model menu",
        "callouts": [
            (1, (60, 105), (102, 105), "Package and resource actions"),
            (2, (60, 229), (102, 229), "Parameter counts"),
            (3, (60, 270), (102, 270), "Training backend"),
        ],
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
        "source": "training-current",
        "crop": (110, 0, 1230, 95),
        "caption": "Training connection and backend configuration",
        "caption_en": "Training connection and backend configuration",
        "callouts": [
            (1, (420, -20), (420, 55), "Backend endpoint"),
            (2, (910, -20), (910, 20), "Connect and check service health"),
            (3, (1218, -20), (1218, 20), "Open separate backend configuration"),
        ],
    },
    "training-settings": {
        "source": "training-current",
        "crop": (110, 100, 1230, 250),
        "caption": "Training settings, optimization and early stopping",
        "caption_en": "Training settings, optimization and early stopping",
        "callouts": [
            (1, (120, 75), (120, 106), "Epochs, batch, learning rate, seed and metric cadence"),
            (2, (85, 176), (110, 176), "Device, precision and optimizer"),
            (3, (1220, 75), (1220, 197), "Weight decay, clipping, warmup, step budget and accumulation"),
            (4, (847, 275), (847, 242), "Early stopping patience and minimum validation improvement"),
        ],
    },
    "parameter-counts": {
        "source": "parameter-counts-current",
        "crop": (290, 145, 1050, 625),
        "caption": "Model parameter counts by node",
        "caption_en": "Model parameter counts by node",
        "callouts": [
            (1, (600, 120), (600, 158), "Total and trainable parameters; analysis completeness"),
            (2, (1075, 330), (1015, 366), "Parameter counts for each node"),
            (3, (930, 655), (930, 613), "Refresh the parameter counts"),
            (4, (265, 551), (510, 551), "Horizontal scrollbar reveals the Message column"),
        ],
    },
    "backend-config": {
        "source": "backend-config-current",
        "crop": (380, 125, 940, 645),
        "caption": "Local backend configuration and profile status",
        "caption_en": "Local backend configuration and profile status",
        "callouts": [
            (1, (970, 241), (920, 241), "Choose the local executor"),
            (2, (970, 343), (750, 343), "Local job root, container runtime and worker image"),
            (3, (350, 430), (660, 443), "Portable JSON profile path"),
            (4, (970, 543), (915, 555), "Local service readiness status"),
            (5, (970, 611), (920, 610), "Safely restart the local backend"),
        ],
    },
    "slurm-configure": {
        "source": "slurm-config-current",
        "crop": (380, 125, 940, 762),
        "caption": "SSH/Slurm executor configuration",
        "caption_en": "SSH/Slurm executor configuration",
        "callouts": [
            (1, (970, 223), (920, 223), "Executor selection"),
            (2, (970, 330), (750, 330), "SSH alias, remote root, SIF image and partition"),
            (3, (970, 486), (750, 486), "CPU, memory, time limit and GPU count"),
            (4, (970, 572), (750, 572), "SSH executable"),
            (5, (970, 628), (790, 622), "Portable JSON profile path"),
            (6, (970, 660), (933, 646), "Visible missing-configuration error"),
            (7, (970, 730), (925, 744), "Safely restart the local backend"),
        ],
    },
    "slurm-dashboard": {
        "source": "training-current",
        "crop": (110, 0, 1230, 250),
        "caption": "Training dashboard with current controls",
        "caption_en": "Training dashboard with current controls",
        "callouts": [
            (1, (420, -20), (420, 55), "Backend endpoint and connection status"),
            (2, (120, 75), (120, 106), "Basic training settings"),
            (3, (85, 176), (110, 176), "Device, precision and optimizer"),
            (4, (847, 275), (847, 242), "Early stopping controls"),
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
    "Backend endpoint and connection status": "Indirizzo del backend e stato della connessione",
    "Choose the local executor": "Scegli l’esecutore locale",
    "CPU, memory, time limit and GPU count": "CPU, memoria, limite di tempo e numero di GPU",
    "Early stopping controls": "Controlli dell’arresto anticipato",
    "Early stopping patience and minimum validation improvement": "Pazienza e miglioramento minimo della validazione per l’arresto anticipato",
    "Epochs, batch, learning rate, seed and metric cadence": "Epoche, batch, tasso di apprendimento, seme e frequenza delle metriche",
    "Horizontal scrollbar reveals the Message column": "La barra orizzontale mostra la colonna Message",
    "Local backend configuration and profile status": "Configurazione del backend locale e stato del profilo",
    "Local job root, container runtime and worker image": "Directory locale dei job, runtime dei container e immagine worker",
    "Local service readiness status": "Stato di disponibilità del servizio locale",
    "Model parameter counts by node": "Conteggio dei parametri del modello per nodo",
    "Parameter counts for each node": "Conteggio dei parametri per ogni nodo",
    "Portable JSON profile path": "Percorso del profilo JSON portabile",
    "Refresh the parameter counts": "Aggiorna il conteggio dei parametri",
    "SSH alias, remote root, SIF image and partition": "Alias SSH, directory remota, immagine SIF e partizione",
    "SSH executable": "Eseguibile SSH",
    "Total and trainable parameters; analysis completeness": "Parametri totali e addestrabili; completezza dell’analisi",
    "Training dashboard with current controls": "Dashboard di addestramento con i controlli attuali",
    "Training settings, optimization and early stopping": "Impostazioni di addestramento, ottimizzazione e arresto anticipato",
    "Visible missing-configuration error": "Errore visibile di configurazione mancante",
    "Weight decay, clipping, warmup, step budget and accumulation": "Decadimento dei pesi, clipping, warmup, budget di passaggi e accumulo",
    "Device, precision and optimizer": "Dispositivo, precisione e ottimizzatore",
    "Package and resource actions": "Azioni per pacchetti e risorse",
    "Parameter counts": "Conteggio dei parametri",
    "Training backend": "Backend di addestramento",
    "Training connection and backend configuration": "Connessione al backend e configurazione dell’addestramento",
    "SSH/Slurm executor configuration": "Configurazione dell’esecutore SSH/Slurm",
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


def write_english_captions(captions: dict) -> None:
    # Keep the compact, one-entry-per-line format used by captions-en.json.
    lines = ["{"]
    entries = list(captions.items())
    for index, (name, value) in enumerate(entries):
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        comma = "," if index + 1 < len(entries) else ""
        lines.append(f"  {json.dumps(name, ensure_ascii=False)}: {encoded}{comma}")
    lines.append("}")
    (OUTPUT_DIR / "captions-en.json").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=tuple(SHOTS),
                        help="generate only the named screenshot stems and update only their captions")
    args = parser.parse_args()
    selected = args.only if args.only else list(SHOTS)
    italian_captions = json.loads((OUTPUT_DIR / "captions.json").read_text(encoding="utf-8"))
    english_captions = json.loads((OUTPUT_DIR / "captions-en.json").read_text(encoding="utf-8"))
    for stem in selected:
        spec = SHOTS[stem]
        calls = make_one(stem, spec)
        key = f"{stem}.png"
        italian_captions[key] = {
            "caption": italian(spec["caption"]),
            "callouts": [
                {"number": call["number"], "region": italian(call["region"])}
                for call in calls
            ],
        }
        if "details" in spec:
            italian_captions[key]["details"] = italian(spec["details"])
        if "caption_en" in spec:
            english_captions[key] = {
                "caption": spec["caption_en"],
                "callouts": [
                    {"number": call["number"], "region": call["region"]}
                    for call in calls
                ],
            }
            if "details_en" in spec:
                english_captions[key]["details"] = spec["details_en"]
    (OUTPUT_DIR / "captions.json").write_text(
        json.dumps(italian_captions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_english_captions(english_captions)


if __name__ == "__main__":
    main()
