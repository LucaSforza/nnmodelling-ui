# Parametri dei pacchetti

I nomi sotto sono le chiavi mostrate in `Inspector > Parameters`. Tutti i pacchetti core elencati sono alla versione `0.1.0` e quindi appaiono con l'identità completa `core.<id>@0.1.0`. `Default non dichiarato` indica che il pacchetto non specifica un valore iniziale nel proprio schema; l'Inspector e il validatore mostrano se quel campo è disponibile e accettabile. I limiti sono quelli dichiarati dallo schema. `top` e `bottom` indicano che il parametro viene anche mostrato nella rispettiva riga della scheda del nodo.

## Parametri core

Ogni parametro occupa una riga: questo rende esplicito lo scopo di ogni campo, compresi quelli che condividono un default o un vincolo.

| Pacchetto | Chiave | Tipo e default | Vincoli, scelte, posizione | Significato |
| --- | --- | --- | --- | --- |
| `core.adaptive-avg-pool2d@0.1.0` | `output_size` | intero, non dichiarato | minimo 1; `bottom` | Dimensione spaziale obiettivo usata dall'adaptive average pooling. |
| `core.add@0.1.0` | — | — | — | Nessun parametro. Somma due o più tensori della stessa forma. |
| `core.batch-norm2d@0.1.0` | `num_features` | intero, non dichiarato | minimo 1; `top` | Numero di canali nella dimensione C dell'input rank-4 NCHW. |
| `core.batch-norm2d@0.1.0` | `eps` | numero, `0.00001` | minimo 0 | Costante di stabilizzazione aggiunta alla varianza durante la normalizzazione. |
| `core.batch-norm2d@0.1.0` | `momentum` | numero, `0.1` | da 0 a 1 | Peso usato nell'aggiornamento della media/varianza mobile. |
| `core.batch-norm2d@0.1.0` | `affine` | booleano, `true` | — | Abilita scala e offset apprendibili dopo la normalizzazione. |
| `core.batch-norm2d@0.1.0` | `track_running_stats` | booleano, `true` | — | Mantiene statistiche mobili di media e varianza per la modalità di valutazione. |
| `core.cast@0.1.0` | `dtype` | dtype, `float32` | `float16`, `bfloat16`, `float32`, `float64`, `int8`, `uint8`, `int16`, `int32`, `int64`, `bool` | Dtype del tensore convertito; la forma rimane invariata. |
| `core.concat@0.1.0` | `dim` | intero, non dichiarato | — | Asse lungo il quale concatenare gli ingressi. |
| `core.conv2d@0.1.0` | `in_channels` | intero, non dichiarato | minimo 1; `top` | Numero di canali attesi in ingresso. |
| `core.conv2d@0.1.0` | `out_channels` | intero, non dichiarato | minimo 1; `bottom` | Numero di canali prodotti dalla convoluzione. |
| `core.conv2d@0.1.0` | `kernel_size` | intero, non dichiarato | minimo 1 | Lato della finestra quadrata di convoluzione. |
| `core.conv2d@0.1.0` | `stride` | intero, `1` | minimo 1 | Passo della finestra tra posizioni adiacenti. |
| `core.conv2d@0.1.0` | `padding` | intero, `0` | minimo 0 | Padding spaziale aggiunto ai bordi dell'input. |
| `core.conv2d@0.1.0` | `dilation` | intero, `1` | minimo 1 | Spaziatura tra gli elementi della finestra. |
| `core.conv2d@0.1.0` | `groups` | intero, `1` | minimo 1 | Numero di gruppi di canali indipendenti nella convoluzione; deve essere compatibile con entrambi i conteggi dei canali. |
| `core.conv2d@0.1.0` | `bias` | booleano, `true` | — | Include un termine di bias apprendibile. |
| `core.cross-entropy@0.1.0` | — | — | — | Nessun parametro. Loss per classificazione; legge il target nominato `target` del batch e produce un risultato di loss scalare. |
| `core.embedding@0.1.0` | `num_embeddings` | intero, non dichiarato | minimo 1; `top` | Numero di indici distinti rappresentabili dalla tabella di embedding. |
| `core.embedding@0.1.0` | `embedding_dim` | intero, non dichiarato | minimo 1; `bottom` | Larghezza del vettore associato a ciascun indice. |
| `core.embedding@0.1.0` | `input_dtype` | dtype, `int64` | `int32`, `int64` | Dtype intero degli indici passati alla tabella. |
| `core.embedding@0.1.0` | `dtype` | dtype, `float32` | `float16`, `bfloat16`, `float32`, `float64` | Dtype dei vettori di embedding. |
| `core.flatten@0.1.0` | `start_dim` | intero, `1` | — | Prima dimensione inclusa nell'intervallo da appiattire. |
| `core.flatten@0.1.0` | `end_dim` | intero, `-1` | — | Ultima dimensione inclusa nell'intervallo da appiattire; `-1` indica l'ultima dimensione. |
| `core.fork@0.1.0` | — | — | — | Nessun parametro. Punto di diramazione che mantiene il tensore in ingresso. |
| `core.horizontal-repeat@0.1.0` | `times` | intero, non dichiarato | minimo 2 | Numero di copie indipendenti della subflow eseguite in parallelo. |
| `core.horizontal-repeat@0.1.0` | `join` | riferimento a `stereotype`; `core.concat@^0.1.0` con `dim: -1` | selettore filtrato ai pacchetti di tipo `join` | Pacchetto che combina le uscite delle copie parallele; scegliendo un altro pacchetto di join si caricano i suoi parametri. |
| `core.input@0.1.0` | `binding` | stringa, non dichiarato | — | Nome dello slot di input del dataset attivo da cui proviene questo input radice. Nelle subflow l'input iniziale è ereditato dal contenitore. |
| `core.layer-norm@0.1.0` | `normalized_shape` | intero, non dichiarato | minimo 1; `top` | Dimensione finale normalizzata per ogni campione. |
| `core.layer-norm@0.1.0` | `eps` | numero, `0.00001` | minimo 0 | Costante di stabilizzazione per il calcolo della varianza. |
| `core.layer-norm@0.1.0` | `elementwise_affine` | booleano, `true` | — | Abilita scala e offset apprendibili per elemento. |
| `core.linear@0.1.0` | `in_features` | intero, non dichiarato | minimo 1; `top` | Larghezza attesa dell'ultima dimensione in ingresso. |
| `core.linear@0.1.0` | `out_features` | intero, non dichiarato | minimo 1; `bottom` | Larghezza dell'ultima dimensione prodotta. |
| `core.linear@0.1.0` | `bias` | booleano, `true` | — | Include un termine di bias apprendibile. |
| `core.linear@0.1.0` | `dtype` | dtype, `float32` | `float16`, `bfloat16`, `float32`, `float64` | Dtype dei pesi e dei calcoli lineari. |
| `core.loss-output@0.1.0` | — | — | — | Nessun parametro. Terminale che raccoglie un singolo risultato di loss; non somma automaticamente più contributi. |
| `core.matmul@0.1.0` | — | — | — | Nessun parametro. Riceve due o più tensori e applica la moltiplicazione matriciale da sinistra a destra. |
| `core.max-pool2d@0.1.0` | `kernel_size` | intero, non dichiarato | minimo 1 | Lato della finestra quadrata di pooling. |
| `core.max-pool2d@0.1.0` | `stride` | intero, `1` | minimo 1 | Passo tra posizioni successive della finestra. |
| `core.max-pool2d@0.1.0` | `padding` | intero, `0` | minimo 0 | Padding aggiunto ai bordi prima del pooling. |
| `core.max-pool2d@0.1.0` | `dilation` | intero, `1` | minimo 1 | Spaziatura tra gli elementi della finestra di pooling. |
| `core.max-pool2d@0.1.0` | `ceil_mode` | booleano, `false` | — | Se attivo, arrotonda per eccesso il calcolo delle finestre; altrimenti usa l'arrotondamento per difetto. |
| `core.mse-loss@0.1.0` | — | — | — | Nessun parametro. Loss di regressione MSE; usa il solo target nominato `target` dichiarato dal pacchetto, con la trasformazione di target specifica `flatten_batch`. Non seleziona né appiattisce target arbitrari. |
| `core.output@0.1.0` | — | — | — | Nessun parametro. Terminale che marca la predizione del modello e riceve un solo arco. |
| `core.positional-encoding@0.1.0` | `d_model` | intero, `512` | minimo 1; `top` | Larghezza della rappresentazione sequenziale cui aggiungere informazione posizionale. |
| `core.positional-encoding@0.1.0` | `max_len` | intero, `5000` | minimo 1; `bottom` | Lunghezza massima della sequenza per cui preparare le posizioni. |
| `core.relu@0.1.0` | `inplace` | booleano, `false` | — | Se attivo, applica la ReLU modificando il tensore d'ingresso in-place. |
| `core.repeat@0.1.0` | `times` | intero, non dichiarato | minimo 1; `top` | Numero di copie indipendenti della subflow applicate in sequenza. |
| `core.scale@0.1.0` | `factor` | numero, `1.0` | `bottom` | Moltiplicatore applicato a ogni valore del tensore. |
| `core.softmax@0.1.0` | `dim` | intero, `-1` | `bottom` | Asse lungo cui normalizzare i valori in probabilità. `-1` indica l'ultima dimensione. |
| `core.subflow-proxy@0.1.0` | — | — | — | Nessun parametro. Delega l'analisi e l'esecuzione a una singola istanza della subflow annidata. |
| `core.transpose@0.1.0` | `dim0` | intero, `-2` | `top` | Prima dimensione da scambiare; `-2` indica la penultima. |
| `core.transpose@0.1.0` | `dim1` | intero, `-1` | `bottom` | Seconda dimensione da scambiare; `-1` indica l'ultima. |

Le scelte dtype sono specifiche del parametro. Ad esempio, `input_dtype` di `Embedding` accetta solo indici interi `int32`/`int64`; il dtype dei vettori è invece limitato ai quattro dtype floating elencati. I vincoli del pacchetto sono verificati quando il campo viene confermato e dal validatore del modello.

## Risorse del progetto mini LLM

Quando si crea `New mini LLM`, il progetto riceve copie modificabili del grafo, del dataset e dei suoi pacchetti. La palette mostra i pacchetti core e i pacchetti personalizzati dichiarati dal progetto; i loro nomi sono descrizioni, mentre `Project resources` e il tooltip mostrano l'identità esatta. Le risorse personalizzate elencate qui sono tutte versione `1.0.0`.

| Identità | Descrizione e campi | Ruolo nell'esempio |
| --- | --- | --- |
| `llm.causal-mask@1.0.0` | `Causal Mask`: parametro `max_length` (intero, default `128`, minimo 1). Maschera le posizioni future nei punteggi di attenzione `[B,T,T]`. | Il valore limita la lunghezza di contesto prevista dal pacchetto; il nodo usa la maschera nel percorso di attenzione. Il parametro si può cambiare nell'Inspector. |
| `llm.token-cross-entropy@1.0.0` | `Token Cross Entropy`: nessun parametro; uscita `loss` di tipo loss; target esterno nominato `target` da `batch.targets.target`. Calcola la cross-entropy media per token da logits `[B,T,V]` a target `[B,T]`. | Definisce la loss di previsione del token successivo. L'Inspector del nodo non offre campi parametro da cambiare; il nodo resta selezionabile, collegabile e rimovibile. |
| `llm.causal-attention@1.0.0` | `Causal Self Attention`: nessun parametro; uscita `out` di tipo output; descrizione legacy: attenzione causale a quattro teste, larghezza modello 64 e larghezza testa 16. | Il pacchetto è conservato tra le risorse del progetto, ma non è usato dal grafo aggiornato: le operazioni di attenzione sono rappresentate da nodi e subflow espliciti. Non c'è un modulo dell'interfaccia per modificare la definizione del pacchetto. |
| `llm.tokens@1.0.0` (dataset) | `Autoregressive token sequences`: input `tokens`, dtype `int64`, forma `[B,128]`; target `target`, dtype `int64`, forma `[B,128]`. | Gli slot sono selezionabili come dataset attivo. `Dataset… > Edit` permette di modificare nome, descrizione e slot; ID/versione sono fissi nel modulo e l'interfaccia non modifica il codice adapter o i campioni. |

L'interfaccia permette di creare nuovi stereotype con `Create stereotype`, non di modificare o eliminare la definizione di uno stereotype già presente. Si possono modificare i parametri delle istanze-nodo nell'Inspector, collegare/rimuovere/spostare nodi e riordinare il grafo. Per le tre risorse LLM sopra, soltanto `Causal Mask` ha parametri modificabili sul nodo. Il dataset manager non modifica l'implementazione Python né i file di dati.

I package JSON descrivono forma e dtype per l'analisi nativa; la presenza di un package Python eseguibile nell'esempio non significa che il client Qt esegua Python durante la modifica del grafo.
