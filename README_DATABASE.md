### Microsoft Security Incident Prediction (GUIDE) Dataset


## Panoramica del dataset (GUIDE)

Questo progetto usa il dataset **Microsoft Security Incident Prediction (GUIDE)** pubblicato su Kaggle per studiare come i dati reali di telemetria di sicurezza possano supportare modelli di **Machine Learning** utili ai **Security Operation Center (SOC)**, soprattutto nella fase di **triage** degli incidenti (cioè classificare e prioritizzare rapidamente ciò che conta davvero). ([kaggle.com][1])

### Perché esiste GUIDE

Nel mondo reale i SOC ricevono un’enorme quantità di segnali e alert ogni giorno. Automatizzare *tutto* è rischioso: un’azione sbagliata può bloccare asset critici. Per questo si punta spesso a sistemi di **risposta guidata** (guided response): invece di “agire automaticamente”, aiutano l’analista a decidere meglio e più velocemente, usando contesto e correlazioni. In questo contesto i prodotti **XDR** sono rilevanti perché aggregano dati da molte fonti (endpoint, rete, cloud, email, identità, SaaS) per rilevare e rispondere alle minacce in tempo quasi reale. ([microsoft.com][2])

### Cosa contiene (in termini pratici)

GUIDE è grande e strutturato: include **evidenze**, **alert** e **incidenti** annotati. In numeri: più di **13 milioni di evidenze** su **33 tipi di entità**, che coprono circa **1,6 milioni di alert** e **1 milione di incidenti annotati**. ([kaggle.com][1])
Inoltre, gli incidenti e i segnali sono collegati a numerosi “detector” e tecniche, includendo riferimenti a **MITRE ATT&CK**, cioè una knowledge base standard di tattiche e tecniche usate dagli attaccanti, basata su osservazioni reali. ([attack.mitre.org][3])

### Gerarchia dei dati: Evidence → Alert → Incident

GUIDE è organizzato su tre livelli, utili per capire “come nasce” un incidente:

1. **Evidence (Evidenza)**
   È il mattone base: un singolo elemento informativo (es. IP, email, utente, host, ecc.) con metadati associati. Da sola spesso non basta a decidere.

2. **Alert**
   Raggruppa più evidenze correlate e segnala un possibile evento di sicurezza. Qui inizia a comparire il contesto (correlazione di segnali).

3. **Incident**
   Raggruppa una o più alert e racconta una “storia” coerente: un possibile scenario di breach o minaccia.

Questa struttura è importante perché consente approcci ML sia **a livello incident**, sia su dati più granulari (alert/evidence), a seconda dell’obiettivo.

### Benchmark e obiettivo principale: predire il triage

Il benchmark principale del dataset è prevedere le decisioni storiche di triage dei clienti, cioè classificare ogni incidente in classi come:

* **TP (True Positive)**: attività realmente malevola.
* **FP (False Positive)**: falso allarme.
* **BP / Benign (Benign True Positive o benigno)**: attività reale ma non malevola (es. test, attività autorizzate). ([learn.microsoft.com][4])

Per questo scopo, Kaggle fornisce un training set (es. `GUIDE_Train.csv`) con feature già pronte per modellare la predizione di triage. La metrica raccomandata è il **Macro F1-score** (utile quando le classi sono sbilanciate o quando vuoi trattare tutte le classi in modo “equilibrato”), insieme a precision e recall. ([scikit-learn.org][5])

### Come lo userò in questo progetto (visione generale)

Nel repository il dataset sarà usato in più modalità, con un taglio “data science end-to-end”:

* **Classificazione**: predire il triage (TP/BP/FP) a livello incidente.
* **Clustering**: scoprire gruppi/pattern ricorrenti di incidenti o alert (es. famiglie di comportamenti, profili di organizzazioni/detector).
* **Regressione**: prevedere grandezze numeriche derivate (es. proxy di impatto, volume atteso, tempi, score continui costruiti nel progetto).
* **Time series / forecasting**: analizzare trend temporali (frequenze di alert/incidenti) e prevedere picchi o cambiamenti.

---

## Glossario (termini chiave)

**Triage**
Processo di valutazione rapida di alert/incidenti per **classificare, validare e prioritizzare** cosa richiede attenzione immediata. ([VMRay][6])

**Machine Learning (ML)**
Insieme di tecniche che permettono a un modello di “imparare” pattern dai dati per fare predizioni o raggruppamenti (es. classificazione, regressione, clustering).

**Etichette di triage (triage labels / etichette di triage)**
Le “classi” assegnate dagli analisti/clienti agli incidenti (es. TP/FP/BP). Sono il **target** per la classificazione supervisionata. ([learn.microsoft.com][4])

**Built-in**
Funzionalità o detector “**integrati di default**” nel prodotto (non creati ad-hoc dall’utente/azienda). In contrapposizione a *custom* (personalizzati).

**MITRE ATT&CK**
Knowledge base standard, pubblica, di tattiche e tecniche degli avversari basata su osservazioni reali; spesso usata per mappare e descrivere comportamenti di attacco. ([attack.mitre.org][3])

**Feature (caratteristica / variabile)**
Colonna/variabile in input al modello (numerica o categoriale) che descrive un incidente/alert (es. conteggi, indicatori, attributi aggregati). Le feature sono ciò che il modello usa per “decidere”.

**Macro F1-score**
Metrica di valutazione per classificazione: calcola F1 per ogni classe e poi fa la media “macro” (ogni classe pesa uguale). Utile quando non vuoi che la classe più frequente domini il risultato. ([scikit-learn.org][5])

