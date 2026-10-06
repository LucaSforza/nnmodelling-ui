# Verifica del manuale

Verifica eseguita il 6 ottobre 2026. Le descrizioni derivano dai contratti della
KB e dall'implementazione corrente; le immagini mostrano il client Qt reale,
controllato con computer use attraverso il bridge noVNC locale.

## Prove eseguite

- Compilazione del client: `CCACHE_DISABLE=1 just build` riuscita.
- Suite core: 13 test su 13 superati; suite GUI: 6 su 6 superati. Le ricette
  `just test` e `just test-ui` hanno inizialmente rilevato l'assenza della directory
  temporanea `/tmp/opencode` richiesta dai fixture. Dopo averla creata, sono stati
  ripetuti i test con `ctest --test-dir build/core --output-on-failure` e
  `ctest --test-dir build/qt -L gui --output-on-failure`.
- Navigazione GUI di menu, scope, inspector, dataset, riferimenti stereotype,
  handle join, diagnostica e pannello Training. Le modifiche dimostrative sono
  state annullate; il progetto usato era una copia locale dell'esempio.
- Backend isolato su loopback, porta 8766, archivio temporaneo dedicato. Health,
  invio GUI, cronologia, metriche e download wheel verificati sul servizio reale.
- Job `86caba53-0690-4c79-b948-4273ed1426b4`: tre epoche, batch 16, learning rate
  0.001, seed 0, pubblicazione ogni step. Completato con 12 aggiornamenti,
  training loss media finale 3.462321, validation loss 3.387799, test loss 3.345268.
- Wheel scaricata dal client: modello caricato senza backend, ciclo d'inferenza
  del tutorial eseguito con successo per 40 caratteri. Per questa verifica la
  wheel è stata estratta in una directory temporanea e importata usando le
  dipendenze del workspace; l'installazione in un nuovo venv resta la procedura
  suggerita al lettore, non una seconda installazione provata qui.
- Esempio API del capitolo server provato su un backend isolato: invio di un
  job breve, completamento e recupero di snapshot, pesi e wheel.
- HTML rigenerato e controllato con `build.py --check`: pagine, link locali,
  ancore e immagini. Entrambe le lingue e il passaggio al medesimo capitolo
  verificati; impaginazione desktop e mobile controllata via computer use.
  La sintassi dei comandi shell e Python è stata controllata anche nella traduzione.

## Limiti osservati

Il percorso con corpus completo e 20 epoche è documentato dai file dell'esempio
e dallo script di preparazione; non è stato rieseguito durante questa sessione.

Il contratto descrive Network 3D, ma questa interfaccia non espone i relativi
controlli: il manuale lo segnala, senza illustrare funzioni non accessibili.

Il modulo stereotype è stato acquisito per intero usando un display virtuale
alto 1600 pixel. Durante le prove di chiusura di questo modulo tramite bridge,
Escape ha terminato la sessione Qt. La creazione e l'annullamento del modulo
sono descritti dal codice; non sono dichiarati come operazioni GUI riuscite.
Il modulo dataset illustrato è quello di modifica: la creazione condivide campi
e tabelle, con le differenze descritte nel testo.

Le prove non hanno modificato gli esempi originali né il modello escluso dalla
richiesta. Screenshot originali, copia dimostrativa e artifact restano fuori dai
file versionati. I servizi temporanei di verifica sono arrestati a fine lavoro.
