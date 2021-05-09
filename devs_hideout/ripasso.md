# Appunti

## SE Block

Paper: <https://arxiv.org/pdf/1709.01507.pdf>

Usato per migliorare la performance *al costo di pochi parametri*.
Composto da:

* Squeeze: Global Average Pooling
* Excitation: un meccanismo di gating con attivazione sigmoide.

Produce dei pesi che modificano l'input tramite semplice moltiplicazione.

## Struttura del modello

Presa dal paper perché stiamo reimplementando

bottleneck_dim=12 b/c facciamo un range di età da 0 a 120

Il modulo context prende 3 input, fa passare ciascuno per il modello
comune, poi concatena i risultati.

Il modulo cascade trasforma quei risultati in una layer nascosta che
rappresenta una distribuzione two-point dell'età. Usa una regolarizzazione
(L1) e un'attivazione softmax.

## Dataprocessing

Un modello stand-alone che preprocessa il dataset e produce un tabellone
in formato pickle utilizzabile dal resto del progetto.

La procedura si compone di due passi:

* "Early dataset", specifica per ogni dataset perché dipende dalla
  formattazione specifica di ciascun dataset
* "True dataset", un passo finale uguale per tutti

### Early dataset

Parsa il filename di ciascuna immagine del dataset per estrarre l'età
di ciascun soggetto, poi crea una tabella intermedia che contiene,
per ciascuna immagine, il suo path e l'età corrispondente.

### True dataset

* Usa opencv per leggere l'immagine da disco
* Riconosce la faccia con la MTCNN (codice esterno)
* Scarta immagini che non hanno esattamente una faccia
  (assegnando un'eta che verrà scartata al passo finale)
* Recuperiamo la regione dove si trova la faccia e i keypoints
  (di fatto ci serve il naso)

  *(Abbiamo bisogno di riformattare la box perché usiamo un
  formato diverso rispetto a quello di col\*i che ha implementato
  la MTCNN)*

* *(abbiamo anche lasciato i fixed_squares, ma solo per ragioni storiche)*
* Una volta completata l'elaborazione, comprime l'immagine in formato JPEG
* Salva l'immagine JPEG e la lista delle crop boxes nel tabellone pandas
* Poi elimina le età fuori dal range [0, 120]
* Limita la tabella alle colonne immagine, età, cropboxes.

## Image manipulation

Parte del processo di caricamento dei dati durante il training.
*(Punto d'ingresso: image_manipulation.get_image_crops)*

### Random erasing

Elimina una regione casuale dell'immagine sostituendola con del rumore.

La regione viene scelta in termini dell'area relativa al totale e
dell'aspect ratio, e uno dei vertici del rettangolo.

Tramite occlusione casuale dell'immagine speriamo di rendere il modello
un po' più robusto.

### Padding

Per evitare problemi in caso le boundingbox finiscano fuori dall'area.
Il tipo di padding scelto "specchia" il bordo.

### Random shift

Sposta ciascuna crop box di un po' in entrambe le direzioni.
Ogni box è trattata indipendentemente.
Se una box supera i limiti dell'immagine (paddata), invece non lo fa.

Spostando la box rispetto alla posizione standard speriamo di
incoraggiare il modello a riconoscere davvero le feature piuttosto che,
boh, guardare in posizioni fisse.

### Ritaglio

Le tre crop box vengono usate per generare altrettante crop dell'immagine,
che vengono ridimensionate a 64x64.

### Altri spostamenti

Applichiamo una rotazione casuale e un flipping rispetto all'asse
vericale, sempre per spostare un po' le feature per incoraggiare il
modello a trovarle veramente.

### Alterazioni dei colori

Applichiamo un normale cambiamento di luminosità e contrasto per
simulare altre condizioni di illuminazione e rendere il modello più
ribusto rispetto a questo.

Per lo stesso motivo applichiamo un cambiamento casuale di luminosità
(codice esterno, adattato per avere un cambiamento casuale).
La procedura consiste nella generazione di due lookup table
(una per incrementare, una per decrementare) in base ai valori
di base dati dallo svizzero e a un numero random *(in modo
non proprio ortodosso a funzionante nei limite che abbiamo dato)*;
dopodiché se riscaldiamo incrementiamo rosso e saturazione e
decrementiamo blu, se raffreddiamo viceversa.

## Training

In train_main carichiamo il dataset preprocessato da disco
e chiamiamo tutti i processi di training che ci interessano.

Gli hyperparametri come lr e alpha (bilancia le loss) sono scelti
secondo il papper.

Il datagen sceglie campioni casuali dal dataset, adattando input e output
in base al livello di ablation selezionato. Fa l'augmentation
descritta sopra (ma solo per il training). Trasforma le età
nella loro rappresentazione two-point secondo un calcolo standard.

Il resto è standard.

## Testing

Fatto sul dataset FGNET.

test_main carica il modello da disco, fa test e calcola le metriche
quali la MAE e la KLD (se applicabile data l'ablation)

Il test ha del boilerplate simile al train, poi effettua le predizioni
dell'intero test set (in batch) e si salva predizioni e groundtruth
per il futuro calcolo delle metriche.

concatenate_batch_result ci trasforma l'output del modello
(una lista di coppie) in una coppia di liste (be', array numpy).
