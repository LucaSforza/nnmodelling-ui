#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["Markdown==3.10.3"]
# ///
"""Render the offline Italian guide; --check detects stale pages and broken links."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import markdown

ROOT = Path(__file__).resolve().parent
CHAPTERS = (
    ("introduction", "index", "Introduzione"),
    ("client", "client", "Usare il client"),
    ("parameters", "parameters", "Catalogo dei parametri"),
    ("server", "server", "Configurare il server"),
    ("tutorial-tiny-llm", "tutorial-tiny-llm", "Tutorial tiny LLM"),
)
DESTINATIONS = {f"{source}.md": f"{target}.html" for source, target, _ in CHAPTERS}


def render(source: str, target: str, label: str, captions: dict) -> str:
    converter = markdown.Markdown(extensions=["tables", "fenced_code", "toc"], output_format="xhtml")
    content = ET.fromstring("<article>" + converter.convert((ROOT / f"{source}.md").read_text()) + "</article>")
    for link in content.iter("a"):
        path, separator, anchor = link.get("href", "").partition("#")
        if path in DESTINATIONS:
            link.set("href", DESTINATIONS[path] + separator + anchor)
    # Standalone images become figures with legends matching the red numbers.
    for parent in list(content.iter()):
        for position, element in enumerate(list(parent)):
            if element.tag != "p" or len(element) != 1 or element[0].tag != "img":
                continue
            image = element[0]
            src = image.get("src", "")
            image_path = (ROOT / src).resolve()
            if not image_path.is_relative_to(ROOT / "assets") or not image_path.is_file():
                raise ValueError(f"Missing or unsafe screenshot: {src}")
            figure = ET.Element("figure", {"class": "annotated"})
            enlarged = ET.SubElement(figure, "a", {"href": src, "class": "image-link", "title": "Apri screenshot a dimensione originale"})
            image.set("loading", "lazy")
            enlarged.append(image)
            legend = ET.SubElement(figure, "figcaption")
            ET.SubElement(legend, "strong").text = image.get("alt", "Screenshot")
            ET.SubElement(legend, "p", {"class": "image-hint"}).text = "Clic sull'immagine per leggere i dettagli a dimensione originale."
            calls = captions.get(Path(src).name, {}).get("callouts", [])
            if calls:
                listing = ET.SubElement(legend, "ol", {"class": "callouts"})
                for call in calls:
                    item = ET.SubElement(listing, "li")
                    ET.SubElement(item, "span", {"class": "badge", "aria-hidden": "true"}).text = str(call["number"])
                    ET.SubElement(item, "span").text = call["region"]
            parent.remove(element)
            parent.insert(position, figure)
    nav = "".join(
        f'<a href="{destination}.html"' + (' aria-current="page"' if destination == target else "") + f'>{html.escape(title)}</a>'
        for _, destination, title in CHAPTERS
    )
    body = ET.tostring(content, encoding="unicode", method="html")
    return f'''<!doctype html>
<html lang="it"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(label)} · NNModelling</title><link rel="stylesheet" href="style.css"/></head>
<body><a class="skip" href="#content">Vai al contenuto</a>
<aside><a class="brand" href="index.html">NN<span>Modelling</span></a>
<p class="edition">Manuale utente · Italiano<br/>Edizione 6 ottobre 2026</p>
<nav aria-label="Capitoli">{nav}</nav><div class="outline"><p>In questa pagina</p>{converter.toc}</div>
<p class="offline">Consultabile offline.<br/><a href="README.md">Sorgenti e verifica</a></p></aside>
<main id="content"><header><span class="eyebrow">DAL GRAFO AL MODELLO</span><span class="chapter">{html.escape(label)}</span></header>
{body}<footer>NNModelling · Manuale italiano · Screenshot dell'applicazione reale</footer></main></body></html>
'''


def verify(pages: dict[str, str]) -> None:
    # HTML serialization uses void tags; parse generated content as HTML.
    from html.parser import HTMLParser

    class References(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()
            self.links = []

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if attributes.get("id"):
                self.ids.add(attributes["id"])
            for key in ("href", "src"):
                if attributes.get(key):
                    self.links.append(attributes[key])

    parsed = {}
    for name, page in pages.items():
        parsed[name] = References()
        parsed[name].feed(page)
    for name, page in parsed.items():
        for ref in page.links:
            if ":" in ref:
                continue
            path, _, anchor = ref.partition("#")
            destination = path or name
            if destination not in pages and not (ROOT / destination).is_file():
                raise ValueError(f"{name}: broken local link {ref}")
            if anchor and destination in parsed and anchor not in parsed[destination].ids:
                raise ValueError(f"{name}: missing anchor {ref}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    captions = json.loads((ROOT / "assets/captions.json").read_text())
    pages = {f"{target}.html": render(source, target, title, captions) for source, target, title in CHAPTERS}
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
