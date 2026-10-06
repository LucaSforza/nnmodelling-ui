#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["Markdown==3.10.3"]
# ///
"""Render both offline editions of the user guide; --check validates pages and local links."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET

import markdown

ROOT = Path(__file__).resolve().parent
CHAPTERS = (
    ("introduction", "index", {"it": "Introduzione", "en": "Introduction"}),
    ("client", "client", {"it": "Usare il client", "en": "Using the client"}),
    ("parameters", "parameters", {"it": "Catalogo dei parametri", "en": "Parameter catalog"}),
    ("server", "server", {"it": "Configurare il server", "en": "Configure the server"}),
    ("tutorial-tiny-llm", "tutorial-tiny-llm", {"it": "Tutorial tiny LLM", "en": "Tiny LLM tutorial"}),
    ("tutorial-vae", "tutorial-vae", {"it": "Tutorial VAE", "en": "VAE tutorial"}),
)
TEXT = {
    "it": {
        "skip": "Vai al contenuto", "edition": "Manuale utente · Italiano", "outline": "In questa pagina",
        "date": "Edizione 6 ottobre 2026",
        "chapters": "Capitoli", "offline": "Consultabile offline.",
        "sources": "Sorgenti e verifica", "image_title": "Apri screenshot a dimensione originale",
        "image_hint": "Clic sull'immagine per leggere i dettagli a dimensione originale.",
        "eyebrow": "DAL GRAFO AL MODELLO", "footer": "NNModelling · Manuale italiano · Screenshot dell'applicazione reale",
        "switch": "Lingua del manuale", "italian": "Italiano", "english": "English",
    },
    "en": {
        "skip": "Skip to content", "edition": "User guide · English", "outline": "On this page",
        "date": "Edition 6 October 2026",
        "chapters": "Chapters", "offline": "Available offline.",
        "sources": "Sources and verification", "image_title": "Open screenshot at original size",
        "image_hint": "Click the image to inspect it at its original size.",
        "eyebrow": "FROM GRAPH TO MODEL", "footer": "NNModelling · English user guide · Screenshots from the real application",
        "switch": "Guide language", "italian": "Italiano", "english": "English",
    },
}


def page_name(language: str, target: str) -> str:
    return "introduction.html" if language == "it" and target == "index" else f"{target}.html"


def render(language: str, source: str, target: str, label: str, captions: dict) -> str:
    source_root = ROOT if language == "it" else ROOT / "en"
    stylesheet = "style.css" if language == "it" else "../style.css"
    strings = TEXT[language]
    destinations = {f"{chapter}.md": page_name(language, destination) for chapter, destination, _ in CHAPTERS}
    converter = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], output_format="xhtml")
    content = ET.fromstring(f"<article>{converter.convert((source_root / f'{source}.md').read_text())}</article>")
    for link in content.iter("a"):
        path, separator, anchor = link.get("href", "").partition("#")
        if path in destinations:
            link.set("href", destinations[path] + separator + anchor)
    for parent in list(content.iter()):
        for position, element in enumerate(list(parent)):
            if element.tag != "p" or len(element) != 1 or element[0].tag != "img":
                continue
            image = element[0]
            src = image.get("src", "")
            image_path = (source_root / src).resolve()
            if not image_path.is_relative_to(ROOT / "assets") or not image_path.is_file():
                raise ValueError(f"Missing or unsafe screenshot: {language}/{src}")
            figure = ET.Element("figure", {"class": "annotated"})
            enlarged = ET.SubElement(figure, "a", {"href": src, "class": "image-link", "title": strings["image_title"]})
            image.set("loading", "lazy")
            enlarged.append(image)
            legend = ET.SubElement(figure, "figcaption")
            caption = captions.get(Path(src).name, {})
            ET.SubElement(legend, "strong").text = image.get("alt", caption.get("caption", "Screenshot"))
            ET.SubElement(legend, "p", {"class": "image-hint"}).text = strings["image_hint"]
            calls = caption.get("callouts", [])
            if calls:
                listing = ET.SubElement(legend, "ol", {"class": "callouts"})
                for call in calls:
                    item = ET.SubElement(listing, "li")
                    ET.SubElement(item, "span", {"class": "badge", "aria-hidden": "true"}).text = str(call["number"])
                    ET.SubElement(item, "span").text = call["region"]
            parent.remove(element)
            parent.insert(position, figure)

    navigation = "".join(
        f'<a href="{page_name(language, destination)}"' + (' aria-current="page"' if destination == target else "") + f'>{html.escape(titles[language])}</a>'
        for _, destination, titles in CHAPTERS
    )
    italian_href = page_name("it", target) if language == "it" else f'../{page_name("it", target)}'
    english_href = f"en/{target}.html" if language == "it" else f"{target}.html"
    language_switch = (
        f'<nav class="language-switch" aria-label="{strings["switch"]}">'
        f'<a lang="it" href="{italian_href}"'
        + (' aria-current="page"' if language == "it" else '') + f'>{TEXT["it"]["italian"]}</a>'
        f'<a lang="en" href="{english_href}"'
        + (' aria-current="page"' if language == "en" else '') + f'>{TEXT["en"]["english"]}</a></nav>'
    )
    body = ET.tostring(content, encoding="unicode", method="html")
    return f'''<!doctype html>
<html lang="{language}"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(label)} · NNModelling</title><link rel="stylesheet" href="{stylesheet}"/></head>
<body><a class="skip" href="#content">{strings["skip"]}</a>
<aside><a class="brand" href="{page_name(language, "index")}">NN<span>Modelling</span></a>
<p class="edition">{strings["edition"]}<br/>{strings["date"]}</p>{language_switch}
<nav aria-label="{strings["chapters"]}">{navigation}</nav><div class="outline"><p>{strings["outline"]}</p>{converter.toc}</div>
<p class="offline">{strings["offline"]}<br/><a href="README.md">{strings["sources"]}</a></p></aside>
<main id="content"><header><span class="eyebrow">{strings["eyebrow"]}</span><span class="chapter">{html.escape(label)}</span></header>
{body}<footer>{strings["footer"]}</footer></main></body></html>
'''


def verify(pages: dict[str, str]) -> None:
    from html.parser import HTMLParser

    class References(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()
            self.links = []
            self.languages = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if attributes.get("id"):
                self.ids.add(attributes["id"])
            for key in ("href", "src"):
                if attributes.get(key):
                    self.links.append(attributes[key])
            if attributes.get("lang") in {"it", "en"}:
                self.languages[attributes["lang"]] = attributes.get("href")

    parsed = {}
    for name, page in pages.items():
        parsed[name] = References()
        parsed[name].feed(page)
    for name, page in parsed.items():
        parent = posixpath.dirname(name)
        chapter = posixpath.basename(name)
        target = "index" if chapter == "introduction.html" else chapter.removesuffix(".html")
        expected_languages = ({"it": page_name("it", target), "en": f"en/{target}.html"} if not parent else
                              {"it": f'../{page_name("it", target)}', "en": chapter})
        if name != "index.html" and page.languages != expected_languages:
            raise ValueError(f"{name}: language switch must link to the matching chapter: {page.languages}")
        for ref in page.links:
            if ":" in ref or ref.startswith("//"):
                continue
            path, _, anchor = ref.partition("#")
            destination = posixpath.normpath(posixpath.join(parent, path)) if path else name
            file_path = ROOT / destination
            if destination not in pages and not file_path.is_file():
                raise ValueError(f"{name}: broken local link {ref}")
            if anchor and destination in parsed and anchor not in parsed[destination].ids:
                raise ValueError(f"{name}: missing anchor {ref}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    captions = {"it": json.loads((ROOT / "assets/captions.json").read_text()),
                "en": json.loads((ROOT / "assets/captions-en.json").read_text())}
    pages = {}
    for language in ("it", "en"):
        for source, target, titles in CHAPTERS:
            name = page_name(language, target) if language == "it" else f"en/{target}.html"
            pages[name] = render(language, source, target, titles[language], captions[language])
    # A relative redirect also works when opening the guide directly from disk.
    pages["index.html"] = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta http-equiv="refresh" content="0; url=en/index.html"/>
<title>NNModelling user guide</title></head>
<body><p><a href="en/index.html">Open the English user guide</a></p>
<p><a href="introduction.html" lang="it">Apri il manuale in italiano</a></p></body></html>
'''
    verify(pages)
    for name, page in pages.items():
        destination = ROOT / name
        if args.check:
            if not destination.is_file() or destination.read_text() != page:
                raise SystemExit(f"Stale page: {destination}; run docs/user/build.py")
        else:
            destination.write_text(page, encoding="utf-8")
    print(f"Verified {len(pages)} pages, local links and screenshots.")


if __name__ == "__main__":
    main()
