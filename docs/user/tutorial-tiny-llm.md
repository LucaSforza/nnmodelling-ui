# Tutorial: addestrare tiny LLM

Questo percorso parte dal modello distribuito in `examples/models/tiny-decoder-llm`, disponibile nel client come **mini LLM**. Crea una copia, controlla dataset e grafo, esegui un training breve e scarica un modello utilizzabile da Python. Il primo esperimento serve a imparare il flusso completo; per una prova più lunga trovi in fondo il percorso con tutto il corpus.

## 1. Preparare client e backend

Dalla root del repository, con compilatori, CMake, Qt 6, `just`, `uv` e un runtime Docker compatibile disponibili:

```sh
just build
just backend-sync
just backend-image
just backend-run
```

Se il backend era già installato e attivo prima dell'aggiornamento, ricostruisci
l'immagine con `just backend-image`, arresta il servizio con Ctrl-C e avvialo di
nuovo con `just backend-run` prima di inviare il job.

Per eseguire questo stesso job sul cluster Sapienza, segui gli illustrati
[passaggi Slurm dalla UI](server.md#eseguire-i-job-sul-cluster-sapienza-dalla-ui).
Scegli l'esecutore con `Configure backend…`; lascia invariati endpoint e
parametri Training qui sotto. La guida mostra anche come preparare immagine
`.sif` e directory remota.

Lascia il terminale del server aperto. In un secondo terminale avvia il client:

```sh
./build/qt/nnmodelling-ui
```

Il backend avviato manualmente ascolta su `http://127.0.0.1:8765`. Negli screenshot di questo tutorial compare **8766**, scelta per isolare la prova da una porta già occupata. Se avvii il backend da `Configure backend…`, l'app sceglie una porta loopback libera e aggiorna `Endpoint` automaticamente. Per riprodurre l'indirizzo illustrato con un server manuale avvia invece:

```sh
NNMODELLING_BACKEND_PORT=8766 just backend-run
```

Usa nel client la porta del server che hai avviato. Se hai scelto un token, inseriscilo nel pannello Training come spiegato nella [configurazione del backend](server.md).

## 2. Creare una copia del modello

Nel selettore iniziale scegli `New mini LLM`, oppure usa `File > New from template > mini LLM…`. Scegli una cartella genitore, inserisci l'ID `llm` e conferma il nome visualizzato. Il programma crea la nuova sottocartella e la apre; una destinazione già esistente viene rifiutata.

![Template mini LLM nel menu File](assets/menu-template.png)

Il progetto originale resta disponibile per altri esperimenti. Se preferisci una copia manuale, copia tutta `examples/models/tiny-decoder-llm` in una cartella di lavoro e apri **quella cartella**, non il solo file `model.json`, con `Open project`.

## 3. Leggere il grafo prima di inviarlo

Il modello usa un vocabolario di **65 caratteri**, contesto **128**, larghezza **64**, **due** blocchi decoder, **quattro** teste di attenzione di larghezza **16** e uno strato feed-forward di larghezza **256**. L'Input radice riceve `tokens`; Embedding e Positional Encoding preparano le rappresentazioni; Repeat applica i blocchi decoder; LayerNorm e Linear producono i logit; Token Cross Entropy alimenta `Loss Output`. `Output` raccoglie la predizione.

![Grafo radice del modello linguistico](assets/overview.png)

Controlla `Project resources`: `llm.tokens@1.0.0` deve essere attivo. `Dataset… > Edit` consente di leggere gli slot senza modificarli:

| Slot | Dtype | Forma | Significato |
| --- | --- | --- | --- |
| Input `tokens` | `int64` | `[B, 128]` | Sequenze di indici dei caratteri. |
| Target `target` | `int64` | `[B, 128]` | La stessa sequenza spostata di un carattere: il simbolo successivo da predire. |

Chiudi con `Cancel` se stai solo ispezionando. Il dataset incluso contiene **64** finestre di training, **16** di validazione e **16** di test, ricavate da un estratto di Tiny Shakespeare. È già pronto: il primo training non richiede download del corpus.

Seleziona Repeat: l'Inspector mostra `times = 2` e uscita `float32[B, 128, 64]`. Apri `Scope` e scegli il blocco decoder; nel suo ambito seleziona HorizontalRepeat: `times = 4`, `join = Concat`, `dim = -1`. Entra nello scope delle teste per vedere `Q × Kᵀ`, scala `0.25 = 1/√16`, maschera causale, Softmax e moltiplicazione per V. Il join concatena quattro risultati larghi 16 in un tensore largo 64.

![Attenzione esplicita dentro HorizontalRepeat](assets/attention.png)

Torna a `Root` dall'albero Scope. L'uscita dei logit deve avere forma `float32[B, 128, 65]`, mentre la loss è scalare. `No model problems` indica che l'analisi delle forme è completa. La correttezza delle forme e la disponibilità delle risorse Python sono entrambe necessarie: la prima si vede nel client, la seconda viene verificata quando il backend esegue il progetto.

## 4. Collegarsi e scegliere un esperimento breve

Premi `Training`. Imposta l'endpoint corretto, lascia vuoto `Bearer token` se il server non lo richiede e premi `Connect / check health`. La riga di stato deve indicare che il runtime è pronto per il training.

![Collegamento al servizio locale](assets/training-connect.png)

Per riprodurre la prova illustrata usa:

| Campo | Valore | Motivo |
| --- | --- | --- |
| `Epochs` | `3` | Tre passaggi sul piccolo dataset. |
| `Batch` | `16` | Quattro aggiornamenti per epoca con 64 finestre di training. |
| `Learning rate` | `0.001` | Passo dell'ottimizzatore Adam. Usa il punto decimale. |
| `Seed` | `0` | Fissa l'inizializzazione e le sorgenti casuali del job. |
| `Publish every N steps` | `1` | Pubblica e valida a ogni aggiornamento; utile per vedere subito le curve del piccolo esperimento. |

![Configurazione del training breve](assets/training-settings.png)

Premi **Save project and submit** una volta. Il client salva le modifiche prima dell'invio, raccoglie risorse e dati e crea il job. Attendi che compaia in `Job history`, poi selezionalo: questa selezione carica curve, stato e metriche. Se resta visibile un vecchio messaggio di invio, `Connect / check health` aggiorna la riga di connessione; lo stato della riga del job e le metriche descrivono l'esperimento.

## 5. Leggere le curve e riconoscere il completamento

La serie blu è la loss di training, quella rossa la loss di validazione. Un valore più basso significa un errore medio minore per questo obiettivo. Il training misura i dati su cui il modello viene aggiornato; la validazione misura dati separati senza aggiornare i pesi. Se il training continua a migliorare e la validazione peggiora, il modello potrebbe adattarsi troppo ai dati di training.

![Job reale completato, curve e test loss](assets/training-dashboard.png)

`Optimizer step` conta gli aggiornamenti globali. In questo esperimento sono **12**: quattro per ciascuna delle tre epoche. I checkbox nascondono o mostrano le serie; `Scale > Log` cambia la rappresentazione della loss e non modifica il job. La test loss viene calcolata alla fine su un terzo insieme separato.

Immagine e metriche sopra documentano il job storico `86caba53-0690-4c79-b948-4273ed1426b4`: è arrivato a `completed`, con training loss media dell'ultima epoca **3.462321**, validation loss **3.387799** e test loss **3.345268**. La schermata di download mostra anche il vecchio nome della wheel. Non sono risultati né un artifact di una nuova esecuzione del percorso con ID `llm`; versione delle dipendenze, runtime e macchina possono influire sui valori. Il piccolo estratto e tre epoche dimostrano il flusso, ma non bastano a promettere testo di buona qualità.

Se il job è `failed`, leggi `Error` nel pannello e il `worker.log` del server. Se vuoi interrompere un job ancora in coda o in esecuzione, selezionalo e premi `Cancel job`. Chiudere il pannello con `Close` lascia il job al server. Le modifiche fatte nel client dopo l'invio non cambiano lo snapshot già in addestramento.

## 6. Scaricare e usare il modello

Con il job `completed` selezionato, premi `Download weights` per ottenere `weights.safetensors` e `Download wheel` per il pacchetto Python. Per il nuovo job creato sopra con ID progetto `llm`, il nome atteso è `nnm_llm-0.1.0-py3-none-any.whl`; `nnm_llm.whl` non è valido perché mancano versione e tag. La distribuzione `nnm_llm` deriva dall'ID del progetto: sequenze di caratteri diverse da lettere e cifre ASCII diventano `_`, gli underscore ai margini sono rimossi e il risultato è minuscolo. Conserva il nome completo proposto per la wheel. Un job storico già salvato può invece scaricare il suo nome precedente: non rinominare né modificare wheel esistenti.

![Scelta del file della wheel esportata](assets/download-wheel.png)

In una nuova cartella prepara un ambiente isolato e installa la wheel scaricata. Sostituisci il percorso con quello reale; `pip` installa anche le dipendenze dichiarate, che devono essere disponibili dalla rete o da una cache locale:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install /percorso/nnm_llm-0.1.0-py3-none-any.whl
```

Il nome della distribuzione (`nnm_llm`) è diverso dal modulo Python, che resta specifico del job e ha forma `nnmodel_<id-job-normalizzato>`. Prima di creare `infer.py`, leggi il nome del modulo dalla wheel scaricata con questo comando, senza importare codice:

```sh
python3 - /percorso/nnm_llm-0.1.0-py3-none-any.whl <<'PY'
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    for name in wheel.namelist():
        if name.startswith("nnmodel_") and name.count("/") == 1 and name.endswith("/__init__.py"):
            print("Modulo da importare:", name.split("/")[0])
PY
```

Il frammento seguente mostra **solo l'import del job storico** nelle schermate e nelle metriche sopra:

```python
import torch
from nnmodel_job_86caba53_0690_4c79_b948_4273ed1426b4 import Model

torch.set_num_threads(2)
model = Model()  # Carica i pesi inclusi nella wheel.
text = "ROMEO:\n"
for _ in range(40):
    prediction = model.inference(text[-128:])
    if not prediction:
        raise RuntimeError("Il modello ha restituito una predizione vuota")
    text += prediction[-1]
print(text)
```

Salva il codice in `infer.py`, sostituendo l'import storico con il nome trovato nella wheel del tuo job; non dedurlo dal nome della distribuzione.

Esegui `.venv/bin/python infer.py`. Il testo passato all'adapter deve contenere caratteri del vocabolario e un contesto non vuoto lungo al massimo 128 caratteri. `infer` è un alias di `inference`. La funzione restituisce una predizione per le posizioni della sequenza; il ciclo usa l'ultimo carattere predetto per costruire una continuazione greedy. Se vuoi pesi alternativi compatibili, costruisci `Model(weights_path="/percorso/weights.safetensors")`.

La wheel contiene il runtime privato necessario: per usarla non servono il repository originale, il dataset di training o una chiamata al server. Job diversi dello stesso progetto, oppure di progetti con ID che producono lo stesso nome normalizzato, condividono distribuzione e versione: installa le rispettive wheel in ambienti separati. Gli artifact restano distinti per job. `examples/implementation/llm` è un consumer configurato per uno specifico job storico a 20 epoche; non cambiarne la wheel esistente.

## 7. Passare al corpus completo

Quando il percorso breve funziona, prepara **un'altra** cartella per il corpus completo. Dalla root del repository:

```sh
uv run --group examples python tools/prepare_full_llm.py \
  --output-project .computer-use/manuale-full-llm
```

La destinazione non deve esistere. Il comando verifica il corpus Tiny Shakespeare e il suo checksum, copia il modello e prepara gli split contigui 80/10/10 prima di estrarre finestre: **6.971** training, **871** validazione, **871** test. Ogni finestra usa 128 caratteri d'ingresso e il carattere successivo come target. Se la sorgente non è già in cache, la preparazione richiede rete; il container di training rimane senza rete.

Apri `.computer-use/manuale-full-llm` con `File > Open project…`. Nel pannello Training inserisci manualmente i valori di `training.json`:

| Campo | Valore corpus completo |
| --- | --- |
| `Epochs` | `20` |
| `Batch` | `64` |
| `Learning rate` | `0.001` |
| `Seed` | `0` |
| `Publish every N steps` | `100` |

Il pannello **non carica automaticamente** `training.json`. Questa configurazione prevede 109 step per epoca e 2.180 in totale; le code finali di epoca vengono pubblicate anche quando non coincidono con il multiplo di 100. Richiede più tempo del tutorial breve. Invia, seleziona il nuovo job e scarica i suoi artifact quando arriva a `completed`.

## 8. Ritrovare l'esperimento e chiudere

La cronologia appartiene alla directory job del backend. Riavviando lo stesso servizio con la stessa `NNMODELLING_JOB_ROOT` ritrovi gli esperimenti. `Restore snapshot…` chiede una cartella genitore e un nome di directory nuovo, ricrea il progetto usato per quel job e lo apre. Ripristina il progetto, non un addestramento da continuare: un nuovo invio inizializza un nuovo job; il pannello non offre resume o fine-tuning da pesi scaricati.

Salva il progetto con `File > Save`, chiudi il client e ferma il backend con **Ctrl-C** nel suo terminale dopo la conclusione dei job. Conserva wheel e pesi dove archivi i risultati degli esperimenti.
