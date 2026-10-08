# Dare forma a una rete

Una rete neurale comincia molto prima del primo aggiornamento dei pesi. Comincia da una domanda: che cosa entrerà nel modello, quali trasformazioni attraverserà, quale risultato dovrà produrre? NNModelling rende visibile questo percorso. Ogni operatore diventa un nodo, ogni collegamento racconta il passaggio di un tensore, ogni sottografo permette di raccogliere una parte del progetto e guardarla da vicino.

Il canvas è un banco di lavoro. Puoi partire da un progetto vuoto, scegliere un componente dalla palette, collegarlo agli altri e cambiarne i parametri nell'Inspector. Oppure puoi aprire un esempio e seguirne il ragionamento: dall'immagine ai dieci logit di un classificatore MNIST, dai caratteri alle predizioni del piccolo modello linguistico. Le forme dei tensori aiutano a leggere il progetto: `[B, 128, 64]` indica un batch di sequenze con 128 posizioni e 64 caratteristiche per posizione. La lettera `B` lascia aperta la dimensione del batch; gli altri numeri descrivono scelte concrete della rete.

Il vantaggio di questa vista emerge quando il modello cresce. Una connessione sbagliata può essere difficile da riconoscere dentro una lunga definizione; qui ha un'origine e una destinazione. Un parametro incoerente compare vicino al nodo che lo usa. Una subflow tiene insieme un blocco senza nasconderne il contenuto: puoi espandere l'anteprima oppure entrare nel suo ambito e lavorare sui singoli operatori. Nel tiny LLM, per esempio, l'attenzione resta leggibile come una sequenza di proiezioni Query, Key e Value, moltiplicazioni, maschera causale e Softmax.

## Dal disegno all'esperimento

NNModelling divide il lavoro fra due strumenti complementari. Il **client** è l'applicazione desktop con cui costruisci e ispezioni il progetto. Verifica collegamenti, parametri e forme dei tensori; conserva il grafo e le risorse nella cartella del progetto. Questa analisi è utile anche mentre la rete è incompleta: puoi continuare a progettare, salvare e correggere i punti indicati in `Model problems`.

Il **server**, chiamato anche backend, riceve una copia del progetto e ne gestisce l'addestramento. Il backend e l'API restano su questo computer; puoi eseguire il worker in un container locale oppure inviare solo i job al cluster Sapienza con Slurm e Singularity. Il dataset fornisce input e target; gli operatori Python eseguono il calcolo numerico; l'ottimizzatore aggiorna i pesi. Nel client segui loss di training e validazione, ritrovi gli esperimenti nella cronologia e scarichi i risultati. Una modifica fatta al grafo dopo l'invio prepara un esperimento futuro: il job già avviato conserva la propria copia immutabile. Configurazione e tutorial sono nella [guida al server](server.md).

Alla fine puoi portare il modello fuori dall'editor. I **pesi** sono un file `safetensors`; la **wheel** è un pacchetto Python che contiene grafo, risorse necessarie all'inferenza, adapter del dataset e pesi addestrati. Una volta installata con le sue dipendenze, espone `Model.infer(...)` e `Model.inference(...)`; include anche i metodi nominati nella sezione `Operations`, per esempio `Model.encode(...)` e `Model.decode(...)`. Il backend può essere spento.

## Le parole che incontrerai

| Termine | Come leggerlo nel software |
| --- | --- |
| **Progetto** | La cartella che raccoglie `model.json`, pacchetti personalizzati e dataset dichiarati. Copiare l'intera cartella conserva il progetto modificabile. |
| **Pacchetto / stereotype** | Un componente disponibile nella palette: definisce tipo, parametri, handle e regole del nodo. I pacchetti core sono distribuiti con il programma; quelli personalizzati appartengono al progetto. |
| **Nodo e handle** | Un'istanza di un componente e i punti da cui entrano o escono i tensori. Un handle d'ingresso occupato non accetta un secondo arco. |
| **Scope / subflow** | Un ambito del grafo e il nodo che ne contiene il sottografo. `Root` è il livello principale. |
| **Dataset attivo** | La risorsa che dichiara input e target e fornisce i batch per training, validazione e test. Il parametro `binding` di un Input radice sceglie il nome dello slot. |
| **Output** | Il terminale della predizione, disegnato come un cerchio marrone. |
| **Loss Output** | Il terminale dell'obiettivo da minimizzare, disegnato come un cerchio rosso. La loss deve essere calcolata e collegata esplicitamente. |
| **Job / snapshot** | Un esperimento e la copia dei file e della configurazione con cui è stato inviato. Ripristinare uno snapshot crea un nuovo progetto. |
| **Epoca / step** | Un passaggio sui dati di training / un aggiornamento dell'ottimizzatore. Una singola epoca può contenere molti step. |

## Un buon primo percorso

Per prendere confidenza, apri una copia di `mini LLM`, osserva il grafo radice e seleziona qualche nodo. Segui le forme nell'Inspector, entra nel blocco decoder e poi nell'attenzione. Quando il percorso ti è chiaro, avvia il backend e invia il piccolo training del [tutorial LLM](tutorial-tiny-llm.md). Per lavorare invece con le rappresentazioni interne di un VAE, segui il [tutorial VAE](tutorial-vae.md): crea `encode` e `decode` dalla UI e richiamali dalla wheel Python.

La [guida al client](client.md) accompagna ogni area della finestra; la [guida al server](server.md) raccoglie configurazione e comandi; il [catalogo dei parametri](parameters.md) serve quando vuoi capire l'effetto di un campo specifico. Puoi leggere queste pagine in ordine oppure tenerle aperte accanto al software e consultarle durante il lavoro.

Questa edizione descrive l'interfaccia Qt verificata il **6 ottobre 2026**. Gli screenshot provengono dall'applicazione reale, pilotata con computer use attraverso il ponte locale noVNC; frecce e numeri sono annotazioni editoriali. Le etichette inglesi restano identiche a quelle dei pulsanti. Il percorso linguistico usa `examples/models/tiny-decoder-llm`; altri modelli sperimentali non fanno parte del tutorial.
