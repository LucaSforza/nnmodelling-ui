# English user guide

Open [index.html](index.html) in a browser. The complete guide works offline without a web server, JavaScript, remote fonts or a CDN. Click images to inspect them at their original size; red arrows and numbers correspond to the adjacent captions. On narrow screens, captions move below the image. Printing uses a dedicated stylesheet.

Sources: [introduction](introduction.md), [client](client.md), [parameters](parameters.md), [server](server.md), [tiny LLM tutorial](tutorial-tiny-llm.md), and [verification notes](verification.md).

To regenerate and verify the HTML from the repository root:

```sh
uv run --script docs/user/build.py
uv run --script docs/user/build.py --check
```

If Python-Markdown 3.10.3 is already installed, use `python3 docs/user/build.py` and the same command with `--check`. Generated HTML is versioned so readers do not need to install build tools. The check also validates local links, anchors and image files.

Screenshots were captured from the real Qt client using computer use through the local noVNC bridge. `annotate.py` makes cropped and annotated copies with Pillow; original captures from that session remain local and ignored in `.computer-use/user-docs/`. Screenshots do not need to be regenerated to read or update the text. [The English captions file](../assets/captions-en.json) maps callout numbers to descriptions. The full stereotype form capture uses a 1600-pixel-high virtual display so its footer is visible.

The [editorial verification notes](verification.md) distinguish performed checks, observed results and environment limits. This guide is descriptive; the KB under `docs/knowledge/` remains the project's normative source.
