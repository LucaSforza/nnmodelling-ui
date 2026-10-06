# NNModelling user guide · Manuale utente

Choose an edition / Scegli un'edizione:

- [English guide](en/index.html)
- [Manuale in italiano](index.html)

## English

The complete guide works offline in a browser without a web server, JavaScript, remote fonts or a CDN. Click an image to inspect it at its original size; red arrows and numbers correspond to the adjacent captions. On narrow screens, captions move below the image. Printing uses a dedicated stylesheet.

Sources: [introduction](en/introduction.md), [client](en/client.md), [parameters](en/parameters.md), [server](en/server.md), [tiny LLM tutorial](en/tutorial-tiny-llm.md), and [verification notes](en/verification.md).

## Italiano

Apri [index.html](index.html) nel browser: il manuale completo funziona offline, senza server web, JavaScript, font remoti o CDN. Le immagini sono cliccabili per leggere il dettaglio originale; frecce rosse e numeri corrispondono alle legende affiancate. Su schermi stretti la legenda passa sotto l'immagine. La stampa usa un foglio di stile dedicato.

Sorgenti: [introduzione](introduction.md), [client](client.md), [parametri](parameters.md), [server](server.md), [tutorial tiny LLM](tutorial-tiny-llm.md) e [verifica](verification.md).

## Build and verification / Generazione e verifica

Run either command from the repository root. If Python-Markdown 3.10.3 is already installed, use `python3` in place of `uv run --script`.

```sh
uv run --script docs/user/build.py
uv run --script docs/user/build.py --check
```

Generated HTML is versioned so readers do not need to install build tools. The check validates all ten pages, local links, anchors and image files. Shared screenshots live in `assets/`; `captions.json` and `captions-en.json` hold the Italian and English callout text. Screenshots were captured from the real Qt client using computer use through the local noVNC bridge. Original captures and working copies remain local and ignored in `.computer-use/user-docs/`. The editorial verification pages distinguish performed checks, observed results and environment limits. The guide is descriptive; the KB under `docs/knowledge/` remains the project's normative source.
