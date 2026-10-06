# Manuale utente italiano

Apri [index.html](index.html) nel browser: il manuale completo funziona offline,
senza server web, JavaScript, font remoti o CDN. Le immagini sono cliccabili per
leggere il dettaglio originale; frecce rosse e numeri corrispondono alle legende
affiancate. Su schermi stretti la legenda passa sotto l'immagine. La stampa usa
un foglio di stile dedicato.

Sorgenti: [introduzione](introduction.md), [client](client.md),
[parametri](parameters.md), [server](server.md),
[tutorial tiny LLM](tutorial-tiny-llm.md).

Per rigenerare e verificare l'HTML dalla root del repository:

```sh
uv run --script docs/user/build.py
uv run --script docs/user/build.py --check
```

Se Python-Markdown 3.10.3 è già installato, puoi usare `python3 docs/user/build.py`
e lo stesso comando con `--check`. Le pagine HTML sono versionate per permettere
la lettura senza installare strumenti. Il controllo verifica anche link locali,
ancore e presenza delle immagini.

Gli screenshot sono acquisiti con computer use dal client Qt reale attraverso
il ponte noVNC locale. `annotate.py` crea copie ritagliate e annotate con Pillow;
gli originali di questa sessione restano locali e ignorati in
`.computer-use/user-docs/`. Non è necessario rigenerare gli screenshot per
leggere o aggiornare il testo. `assets/captions.json` mantiene la corrispondenza
fra numeri e descrizioni. La cattura completa del modulo stereotype usa uno
schermo virtuale alto 1600 pixel per includere il footer.

La [verifica editoriale](verification.md) distingue prove eseguite, risultati
osservati e limiti dell'ambiente. Il manuale è descrittivo; la KB sotto
`docs/knowledge/` resta la fonte normativa del progetto.
