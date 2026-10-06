# Guida al client NNModelling

Questa guida descrive i controlli del client Qt e il loro effetto sul progetto. Le etichette mostrate nell'interfaccia sono riportate in inglese, come appaiono a schermo. Le modifiche al grafo diventano parte del progetto con `File > Save`; la creazione di risorse e la modifica dei metadati di un dataset salvano invece il progetto al termine dell'operazione.

![Finestra principale e aree di lavoro](assets/overview.png)

## Avvio e progetti

Nell'avvio desktop normale compare il selettore progetti. `New project` crea un progetto vuoto; `New MNIST MLP` e `New mini LLM` creano copie modificabili dei rispettivi esempi; `Open project` apre una cartella di progetto; `Close` chiude il selettore e lascia aperta la finestra principale. La finestra principale può quindi restare aperta senza un progetto. Se l'applicazione viene avviata tramite bridge di automazione senza percorso progetto, il selettore iniziale è saltato: per aprirlo si usa `File > New project…`, `File > Open project…` oppure `Training`. Le copie dei modelli di esempio sono indipendenti dagli originali.

La creazione chiede prima la cartella genitore, poi `Model ID` e `Display name`. L'ID determina il nome della nuova cartella. Per i modelli iniziali i campi partono rispettivamente da `mnist-mlp` / `MNIST MLP` o `mini-llm` / `mini LLM`; per un progetto vuoto l'ID parte vuoto e il nome propone l'ID inserito. La versione del progetto non è un campo modificabile in questa finestra. Una cartella di destinazione già esistente o un ID non valido produce un messaggio e non sostituisce il progetto attivo.

`File` contiene:

| Voce | Effetto |
| --- | --- |
| `New project…` (`Ctrl+N`) | Avvia la creazione di un progetto vuoto. |
| `New from template > MNIST MLP…` | Chiede ID e nome per una copia dell'esempio classificatore MNIST. |
| `New from template > mini LLM…` | Chiede ID e nome per una copia dell'esempio linguistico mini LLM. |
| `Open project…` (`Ctrl+O`) | Sceglie una cartella contenente il progetto. Prima di sostituire modifiche non salvate chiede se salvarle, scartarle o annullare. |
| `Save` (`Ctrl+S`) | Salva il progetto corrente. Un errore mantiene aperto il progetto e mostra un dialogo. Senza progetto aperto mostra `No project is open`. |
| `Close project` (`Ctrl+F4` su Linux) | Salva, scarta o annulla la chiusura se ci sono modifiche; al termine torna al selettore. La scorciatoia dipende dalla piattaforma. |

Anche chiudendo la finestra principale con modifiche non salvate compare la scelta `Save`, `Discard` o `Cancel`. Annullare lascia il progetto aperto. Il titolo della finestra mostra `*` quando il progetto ha modifiche non salvate.

![Menu File](assets/menu-file.png)

![Selettore iniziale dei progetti](assets/project-chooser.png)

![Identificatore della nuova cartella](assets/project-id.png)

![Nome visualizzato del progetto](assets/project-name.png)

![Scelta del modello iniziale](assets/menu-template.png)

## Barra dei menu e cronologia

`Edit` contiene `Undo` (`Ctrl+Z`) e `Redo` (`Ctrl+Y` sulla piattaforma osservata). Le voci sono abilitate solo quando la cronologia del progetto ha rispettivamente un'azione annullabile o ripristinabile. Annullare/ripristinare aggiorna canvas, selezione, inspector e problemi. Salvataggio e cronologia sono distinti: il titolo perde l'asterisco quando lo stato torna alla revisione salvata.

`Model` contiene `Create stereotype…`, `Create dataset…`, `Manage datasets…` e `Training backend…`. Le prime tre operazioni richiedono un progetto: senza progetto il client mostra il suggerimento di aprirne uno. `Training backend…` apre il pannello anche senza progetto, per controllare il servizio e consultare i job.

`View` contiene `Fit graph` (`Ctrl+0`), `Zoom in` (`Ctrl++` sulla piattaforma osservata), `Zoom out` (`Ctrl+-`) e `Arrange > Vertical/Horizontal`. `Fit graph` inquadra l'ambito corrente e le sue connessioni senza spostare i nodi. Zoom usa il puntatore quando si usa la rotella; i comandi da menu e toolbar scalano intorno al centro del canvas. La rotella del mouse ingrandisce/riduce. Il tasto centrale, oppure `Space` tenuto premuto insieme al tasto sinistro, trascina il canvas. `Escape` annulla un collegamento in preparazione o un trascinamento. Le scorciatoie di zoom possono variare con la piattaforma e il layout della tastiera.

![Cronologia Undo e Redo](assets/menu-edit.png)

![Menu Model](assets/menu-model.png)

![Menu View](assets/menu-view.png)

## Area di lavoro del grafo

La finestra principale è organizzata in tre colonne: palette `Packages` a sinistra, grafo al centro e pannelli `Inspector`, `Project resources` e `Model problems` a destra. I separatori tra colonne e pannelli destri si possono trascinare per cambiare lo spazio disponibile.

### Palette e creazione dei nodi

`Search packages` filtra i pacchetti attivi per nome, ID, versione e tipo; il testo non distingue maiuscole e minuscole. I risultati sono raccolti sotto intestazioni per tipo (`input`, `layer`, `join`, `loss`, `output`, `loss-output`, `subflow` o altra categoria presente). Selezionare una voce foglia aggiorna la selezione della palette; doppio clic oppure `Add to graph` aggiunge quel pacchetto all'ambito corrente, centrato sulla vista. Selezionare l'intestazione di gruppo non aggiunge un nodo. Il colore della voce riprende quello del pacchetto. Senza progetto aperto la palette è vuota; anche un progetto nuovo vuoto ha comunque il catalogo core e mostra i relativi pacchetti.

I nodi si selezionano con clic; trascinando una zona vuota si selezionano più elementi. I modificatori di selezione Qt consentono di aggiungere o rimuovere elementi dalla selezione. Trascinare un nodo lo sposta; per spostarne più di uno insieme, selezionarli e trascinarne uno. Le posizioni si agganciano alla griglia. Selezionare un arco e premere `Delete` o `Backspace` lo rimuove; gli stessi tasti rimuovono i nodi selezionati e i loro archi incidenti. Una subflow con figli non si può eliminare finché il suo contenuto non è stato rimosso. Ogni rifiuto mostra un dialogo con la ragione.

Per collegare due nodi, trascinare da un handle di uscita fino all'handle d'ingresso desiderato. Il collegamento tratteggiato è una bozza; il modello lo accetta solo se direzione, handle, ambito e regole di occupazione/ciclo sono validi. Un collegamento rifiutato mostra `Operation rejected` e non modifica il grafo. Le connessioni permanenti seguono percorsi ortogonali. Il colore distingue il flusso di output normale da quello di loss. I tooltip degli handle di uscita riportano ID e tipo.

I nodi `Input`, `Output` e `Loss Output` sono cerchi terminali con etichetta esterna. Un `Output` raccoglie una predizione; un `Loss Output` raccoglie un risultato di loss; ognuno accetta un solo collegamento. Il modello radice completo prevede un `Output` e un `Loss Output`; un progetto incompleto resta comunque modificabile e salvabile. Un nodo di tipo `join` è mostrato come barra di giunzione. Accanto alla barra `+` aggiunge un handle d'ingresso libero; `−` rimuove l'handle libero più alto. Rimangono visibili almeno due ingressi e tutti quelli già collegati. `−` è disabilitato se toglierebbe un ingresso occupato; `+` è disabilitato al limite di 128 ingressi mostrati.

![Controlli della join](assets/join-controls.png)

![Ingresso aggiuntivo della join](assets/join-extra.png)

### Ambiti e subflow

La barra `Scope` apre l'albero degli ambiti. `Root` mostra i nodi principali; scegliere una subflow mostra il suo ambito immediato. `Orphan scopes` raccoglie ambiti importati senza contenitore raggiungibile, che restano selezionabili. La navigazione mostra un ambito alla volta e non sposta né modifica i nodi.

Una subflow presenta un riquadro con conteggio dei figli. Il footer `Subflow · N nodes ›` si clicca per espandere un'anteprima in sola lettura dentro il canvas; `▾` la richiude. Nell'anteprima si possono passare i puntatori sui nodi, ma non selezionarli, spostarli, collegarli o modificare gli ingressi della join. Un doppio clic sulla subflow oppure il menu contestuale del tasto destro > `Enter subflow` entra nel suo ambito. Lo stesso menu offre `Expand preview` / `Collapse preview`. La freccia di navigazione dell'albero consente di tornare a `Root` o a un ambito antenato. L'espansione è solo una vista temporanea; i nodi figli conservano le proprie coordinate nell'ambito.

![Albero degli ambiti](assets/scope-tree.png)

![Menu contestuale della subflow](assets/subflow-menu.png)

![Ambito del blocco decoder](assets/decoder.png)

### Barra grafico e direzione

La toolbar `Workspace` offre `Scope`, `Training`, `Fit`, `+`, `−` e `Arrange`. `Training` apre lo stesso pannello di `Model > Training backend…`. `Fit` inquadra il contenuto; `+` e `−` ingrandiscono/riducono. `Arrange` applica l'ordinamento verticale predefinito. La freccia del pulsante apre `Vertical` o `Horizontal`; le stesse scelte si trovano in `View > Arrange`. L'ordinamento ricalcola le posizioni dei nodi nell'ambito corrente, aggiorna il progetto in memoria e lo marca come modificato, quindi inquadra il risultato. Per renderlo persistente su disco usare `File > Save`. Nella direzione verticale il flusso va dall'alto in basso; nella direzione orizzontale va verso sinistra. Cambiare direzione è una scelta di presentazione della sessione; usare `Arrange` è l'azione che riordina e registra le posizioni.

![Scelta della direzione di ordinamento](assets/menu-arrange.png)

## Inspector

Scegliere un nodo sul canvas riempie la tabella `Inspector` con colonne `Property` e `Value`. Senza nodo selezionato il pannello è vuoto. `Name` è un campo modificabile: premere Invio o uscire dal campo applica il nuovo nome. `Package` mostra `id@version`; passare il puntatore sulla riga mostra la descrizione del pacchetto.

Quando l'analisi riesce, `Successful outputs` elenca ogni handle con il suo tipo, dtype e forma. Per i terminali compare invece `Consumed tensor` con dtype e forma del tensore ricevuto. Questi valori descrivono l'analisi di forma; non indicano che il modello sia stato addestrato.

Per una `Output` o `Loss Output` dentro una subflow, `Boundary mapping` permette di associare il terminale a un'uscita dichiarata dal contenitore. L'elenco contiene solo uscite dello stesso tipo; `(unmapped)` appare se non c'è ancora un'associazione. Un'associazione importata che non corrisponde più è mostrata come `Invalid mapping: …` per poterla correggere.

`Parameters` contiene un controllo per ogni parametro definito dal pacchetto:

- i parametri con scelte predefinite sono menu a discesa;
- i valori booleani sono caselle da spuntare;
- interi, numeri, stringhe, dtype e valori JSON sono campi di testo, validati dal pacchetto quando si conferma il campo;
- i parametri posizionati `top` o `bottom` possono comparire anche come righe sintetiche sulla scheda del nodo.

Per significato, default, limiti e scelte di ogni campo dei pacchetti distribuiti, consulta il [catalogo dei parametri](parameters.md).

Un parametro di tipo `stereotype` è composto da un selettore di pacchetto e dai campi dello schema di quel pacchetto. Il selettore mostra nome e identità esatta `id@version`, applica l'eventuale filtro per tipo, e ` (choose package)` rappresenta l'assenza di scelta. Scegliere un pacchetto inizializza i suoi parametri ai valori predefiniti; un riferimento già presente mantiene i propri valori. I campi interni usano caselle, menu o testo secondo il tipo. La selezione e ogni campo aggiornato vengono verificati dal progetto; se un valore non è valido il client segnala il rifiuto e conserva i valori confermati. I dati dell'Inspector si aggiornano dopo una modifica.

![Inspector con un nodo selezionato](assets/inspector.png)

![Scelta di un riferimento stereotype](assets/inspector-reference-menu.png)

![Parametri del pacchetto referenziato](assets/inspector-reference.png)

![Associazione del terminale a un handle esterno](assets/inspector-boundary.png)

## Risorse del progetto e selezione dataset

`Project resources` mostra due gruppi espandibili:

- `Datasets` elenca nome e `id@version`. Un segno `✓` indica il dataset attivo. Le righe figlie mostrano `Input: nome [dtype]` e `Target: nome [dtype]`. Selezionare la riga del dataset lo rende attivo; la selezione aggiorna l'analisi e il progetto diventa modificato. Cliccare una riga informativa figlia mostra il percorso risorsa nella barra di stato.
- `Packages` elenca i pacchetti attivi con identità esatta. Espandere un pacchetto mostra le dipendenze nel formato `Requires id vincolo-versione`. Le righe sono informative; non attivano/disattivano risorse.

`New stereotype` apre il creatore di stereotipi; `New dataset` apre il creatore dei dataset; `Dataset…` apre il gestore. Le prime due azioni sono disponibili anche in `Model`.

### Gestore dei dataset

`Dataset…` / `Manage datasets…` apre `Project datasets`, con un elenco a selezione singola. Ogni riga riporta `id@version — nome`; il dataset attivo è marcato `✓ (active)`. `New dataset` apre la scheda vuota. `Edit` apre la scheda del dataset selezionato; il doppio clic su una riga esegue `Edit`. `Select` attiva il dataset selezionato ed è disabilitato se è già attivo. `Edit` e `Select` sono disabilitati quando non c'è una riga selezionata. `Close` chiude il gestore senza ulteriori modifiche.

![Gestore dei dataset](assets/dataset-manager.png)

### Scheda nuovo/modifica dataset

`Create dataset — saves project` contiene i campi `ID`, `Version`, `Name`, `Description` e la casella `Select this dataset after creation` (selezionata inizialmente). La scheda di modifica si intitola `Edit dataset — saves project`: `ID` e `Version` sono bloccati, la casella di selezione non viene mostrata.

Le tabelle `Input slots` e `Target slots` hanno le colonne `Slot name`, `Dtype`, `Shape`. Per ogni riga indicare un nome univoco, un dtype tra `float16`, `bfloat16`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `bool` e dimensioni separate da virgola. La dimensione simbolica `B` rappresenta il batch; le altre devono essere interi positivi. La forma può avere da 1 a 64 dimensioni. Ogni nuova riga parte da dtype `float32` e forma `B`. `Add row` aggiunge una riga; `Remove row` rimuove solo la riga selezionata. È obbligatorio almeno uno slot di input; gli slot target possono essere vuoti.

`Create and save project` crea la risorsa e salva il progetto; in modifica il pulsante è `Save changes`. `Cancel` chiude senza applicare i campi. Gli errori appaiono nella scheda, che rimane aperta per correggere i dati. Modificare un dataset aggiorna solo nome, descrizione e slot mostrati: ID/versione, adapter Python, dati e metadati non presentati dalla scheda sono mantenuti.

![Scheda dataset con campi e tabelle degli slot](assets/dataset-form.png)

![Tabella degli slot target e scelta dtype](assets/dataset-targets.png)

## Creare uno stereotype

`Create stereotype — saves project` si apre da `Model > Create stereotype…` o `New stereotype`. I campi iniziali sono `ID`, `Version` (predefinita `1.0.0`), `Name`, `Description`, `Kind` e `Color` (predefinito `#6b8fc4`). `Kind` offre `input`, `layer`, `join`, `loss`, `output`, `loss-output` e `subflow`.

I metadati hanno significati distinti sia nel modulo stereotype sia nel modulo dataset:

| Campo | Descrizione |
| --- | --- |
| `ID` | Identità tecnica della risorsa: i riferimenti del progetto usano questo valore insieme alla versione. Deve rispettare la sintassi accettata dal creatore. |
| `Version` | Versione della risorsa; permette di distinguere definizioni diverse dello stesso ID. |
| `Name` | Nome leggibile mostrato nei cataloghi e negli elenchi. |
| `Description` | Testo che spiega uso e funzione della risorsa. |
| `Kind` | Ruolo del nodo nel grafo: determina rappresentazione e regole dei collegamenti. |
| `Color` | Colore della voce e della scheda; usa un valore esadecimale come `#6b8fc4`. |

| Scelta `Kind` | Uso |
| --- | --- |
| `input` | Sorgente dei tensori d'ingresso dell'ambito. |
| `layer` | Trasformazione del tensore, come un livello neurale. |
| `join` | Combina più ingressi: il canvas offre gli handle dinamici. |
| `loss` | Calcola un obiettivo, esposto come uscita di tipo loss. |
| `output` | Terminale che consuma la predizione. |
| `loss-output` | Terminale che consuma l'obiettivo di loss. |
| `subflow` | Contenitore di un grafo interno navigabile. |

`Use explicit output handles instead of defaults` attiva la tabella `Output ID` / `Type (output or loss)`. `Add output row` crea una riga (tipo iniziale `output`); `Remove output row` toglie la riga selezionata. Una definizione esplicita sostituisce le uscite implicite. Senza override la didascalia mostra le uscite predefinite: una normale `out` di tipo `output`, l'handle `loss` di tipo `loss` per kind `loss`, nessuna uscita per `output` e `loss-output`. Sono ammesse al massimo un'uscita per tipo, con ID non vuoto e univoco; i terminali non dichiarano uscite.

`Key` è il nome tecnico del parametro che compare nell’Inspector; `Default` è il valore iniziale dei nuovi nodi; `Minimum` limita inferiormente i valori numerici. `Type` determina il controllo e la validazione, mentre `Choices` limita i valori accettati al menu indicato.

`Parameters` è una tabella con colonne `Key`, `Type`, `Default`, `Minimum`, `Choices (comma separated)`, `Position` e una colonna senza intestazione visibile. I tipi sono `boolean`, `integer`, `number`, `string`, `dtype` e array `json`. Il default deve avere il tipo corrispondente; `Minimum` deve essere finito. Le scelte sono valori separati da virgola e fanno comparire un menu nel futuro inspector del nodo. `Position` può essere vuota, `top` o `bottom`: questi due valori mostrano chiave e valore sulla scheda del nodo. `Add parameter row` aggiunge una riga iniziale con tipo `number` e valore `0`; `Remove selected row` rimuove la riga selezionata.

`Dependencies` contiene `Package ID` e `Version constraint`. `Add dependency` aggiunge una riga con vincolo iniziale `0.1.0`; `Remove dependency` toglie la riga selezionata. Gli ID e i vincoli devono essere non vuoti e gli ID non duplicati. Il progetto deve poter risolvere tutte le dipendenze.

La sezione `Optional Lua inference` contiene l'editor sorgente. Il testo iniziale è una regola pass-through esplicita: restituisce il primo input e segnala se manca. Può essere sostituito con una regola Lua valida; il client mostra gli errori accanto all'editor e, quando disponibile, evidenzia la riga. `Create and save project` valida e salva stereotype e progetto; `Cancel` annulla la scheda. Un errore di campo o Lua lascia aperti i valori inseriti per la correzione.

![Scheda di creazione stereotype completa](assets/stereotype-form.png)

![Metadati, uscite e parametri dello stereotype](assets/stereotype-form-top.png)

![Dipendenze, Lua e conferma dello stereotype](assets/stereotype-form-bottom.png)

## Problemi del modello

`Model problems` mostra i problemi dell'analisi di forma; non elenca righe di successo. Ogni riga riporta categoria, nome nodo, ambito e ragione. Le categorie distinguono `Lua compilation error`, `Model error`, `Incomplete` e `Analysis unavailable` con icona, testo e colore. Le cause principali sono raggruppate; le discendenze bloccate sono righe figlie espandibili. `Technical details`, quando presente, apre codice stabile, file/riga sorgente o dettagli tecnici.

`Current scope only` filtra ai problemi dell'ambito mostrato e mantiene visibile come contesto una causa fuori da quell'ambito quando serve. Cliccare un problema navigabile seleziona e centra il nodo, passando al suo ambito. Il problema `Root` segnala la mancanza o invalidità dei terminali principali e non naviga a un nodo. Se l'analisi non è riuscita, `Analysis unavailable` resta visibile con dettagli tecnici espandibili; non viene presentato come un modello senza problemi. Un nodo con problemi mostra anche un piccolo indicatore sul canvas.

![Problemi del modello e filtro per ambito](assets/model-problems.png)

## Training backend

`Training` nella toolbar e `Model > Training backend…` aprono `Training backend`. Il pannello è utilizzabile per collegarsi e consultare job anche senza un progetto; inviare un job richiede invece un progetto valido, che viene salvato prima dell'invio.

In alto, `Endpoint` parte da `http://127.0.0.1:8765`. `Bearer token` è facoltativo e nascosto mentre si digita; rimane in memoria per la sessione. `Connect / check health` controlla servizio e runtime container; `Refresh jobs` aggiorna la cronologia. La riga di stato comunica il risultato in forma leggibile.

I campi di invio sono `Epochs` (10), `Batch` (32), `Learning rate` (0.001), `Seed` (0) e `Publish every N steps` (10). Epochs: 1–10000; batch: 1–4096; learning rate: maggiore di zero e fino a 1; seed: da −2147483648 a 4294967295; pubblicazione: 1–100000. La cadenza regola pubblicazione dei punti e validazione. `Save project and submit` salva prima lo stato corrente e poi invia uno snapshot immutabile; se il salvataggio o l'invio fallisce, il messaggio appare nella riga di stato.

`Job history` mostra data/ora abbreviata, suffisso dell'ID e stato; il tooltip espone ID e data completi. Selezionare un job carica i suoi dettagli e le curve. L'elenco si aggiorna automaticamente ogni 2,5 secondi. `Learning curves` visualizza le serie `Training loss` e `Validation loss`, entrambe inizialmente attive, sullo stesso asse loss. `Scale` sceglie `Linear` o `Log`. L'asse orizzontale è `Optimizer step` per i nuovi punti pubblicati e `Epoch` per metriche storiche prive di punti step. `Final test loss` resta `pending` finché il job non termina con un valore valido.

`Training and worker log` riporta stato, eventuale errore, metriche e test loss conclusiva. I controlli in basso sono `Download weights`, `Download wheel`, `Restore snapshot…`, `Cancel job` e `Close`. I download richiedono un job completato e salvano il file scelto con la finestra di sistema. `Restore snapshot…` ricrea una copia in una nuova cartella scelta dall'utente, la apre come progetto e non sovrascrive una cartella esistente. `Cancel job` invia la richiesta di annullamento del job selezionato. `Close` chiude il pannello.

![Dashboard di training](assets/training-dashboard.png)

![Scelta della scala delle curve](assets/training-scale.png)

## Funzioni non presenti nell'interfaccia corrente

Il progetto descrive un esploratore `Network 3D`, ma la finestra Qt attuale non contiene una scheda, un menu o un comando 3D. Di conseguenza in questa versione non sono disponibili i controlli di volo, visualizzazione, selezione occorrenze, `Fit` o `Home` 3D. Il canvas 2D resta l'unico editor grafico esposto.
