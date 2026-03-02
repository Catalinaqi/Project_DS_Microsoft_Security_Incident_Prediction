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

### Come sarà usata in questo progetto (visione generale)

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


**XDR (Extended Detection and Response)**
Categoria di soluzioni di sicurezza che **correlano e analizzano dati** provenienti da più fonti (endpoint, rete, cloud, email, identità, ecc.) per **rilevare minacce** e **supportare la risposta** in modo più completo rispetto a strumenti isolati.

**Telemetria (Telemetry)**
In cybersecurity indica i **dati raccolti automaticamente** dai sistemi e dagli strumenti di sicurezza: log, eventi, segnali, metriche, attività di rete, processi su endpoint, accessi, email, ecc. Serve per capire “cosa sta succedendo” e ricostruire contesto.

**SOC (Security Operations Center)**
Il team/centro operativo che monitora la sicurezza di un’organizzazione. Si occupa di **rilevare, analizzare, triagiare e gestire** incidenti (spesso 24/7), usando piattaforme SIEM/XDR e procedure operative.

**Asset**
Qualsiasi **risorsa di valore** per l’organizzazione che va protetta: server, PC, account, database, applicazioni, infrastruttura cloud, servizi email, reti, dati sensibili, ecc. Un “asset critico” è un elemento che, se compromesso o bloccato, può causare impatti gravi sul business.

**Risposta guidata (Guided Response)**
Approccio/sistema che **non automatizza completamente** la remediazione, ma **supporta l’analista** passo-passo con **raccomandazioni basate sul contesto** (dati correlati, storico, priorità, indicatori di compromissione) per decidere **quali azioni fare**, in che ordine e con quale urgenza.

In pratica: invece di “chiudere automaticamente” un incidente, la risposta guidata propone azioni tipo:

* verifiche consigliate (cosa controllare per confermare la minaccia),
* azioni di contenimento (isolamento host, blocco IP, reset credenziali),
* azioni di remediation (rimozione persistenza, patch, regole, cleanup),
* note su rischi/impatti (evitare blocchi su asset critici).

**Data Dictionary (GUIDE_Train) — con livello gerarchico**
| #  | Colonna              | Tipo (DuckDB)                | Completezza | HierarchyLevel        | Key / Aggregation                         | Descrizione e utilità |
| -- | -------------------- | ---------------------------- | ----------- | --------------------- | ------------------------------------------ | --------------------- |
| 1  | Id                   | BIGINT                       | 100%        | INCIDENT (case id)     | (OrgId, IncidentId)                        | Identificatore del **caso/incidente** (case-level). **Non è univoco per riga**: si ripete su più record perché un incidente include più alert e più evidenze. Utile come ID alternativo del caso. |
| 2  | OrgId                | BIGINT                       | 100%        | ALL                   | (OrgId, …)                                 | ID organizzazione/tenant. Utile per segmentazione, split stratificati e analisi per cliente. |
| 3  | IncidentId           | BIGINT                       | 100%        | INCIDENT              | (OrgId, IncidentId)                        | Identificativo dell’incidente (caso). Raggruppa una o più alert. Base per modellare triage a livello incidente. |
| 4  | AlertId              | BIGINT                       | 100%        | ALERT                 | (OrgId, AlertId)                           | Identificativo dell’alert all’interno dell’incidente. Un alert può avere più evidenze (più righe). |
| 5  | Timestamp            | TIMESTAMP WITH TIME ZONE     | 100%        | ALERT/EVIDENCE         | group-by time / order-by time              | Momento dell’evento. Fondamentale per ordering, sequenze e analisi temporali (trend/time series). |
| 6  | DetectorId           | BIGINT                       | 100%        | ALERT                 | (OrgId, AlertId)                           | ID del detector/regola che ha generato l’alert. Utile per pattern e stabilità/qualità del segnale. |
| 7  | AlertTitle           | BIGINT                       | 100%        | ALERT                 | (OrgId, AlertId)                           | Titolo/descrizione dell’alert codificato (anonimizzato, non testo). Feature categoriale. |
| 8  | Category             | VARCHAR                      | 100%        | ALERT                 | (OrgId, AlertId)                           | Categoria dell’alert (es. InitialAccess, Exfiltration). Feature chiave per classificazione/clustering. |
| 9  | MitreTechniques      | VARCHAR                      | 44%         | ALERT                 | (OrgId, AlertId)                           | Tecniche MITRE ATT&CK (multi-valore). Spesso mancante; utile per interpretabilità e feature engineering. |
| 10 | IncidentGrade        | VARCHAR                      | 100%        | INCIDENT (label)      | (OrgId, IncidentId)                        | **Etichetta di triage** dell’incidente (TruePositive / FalsePositive / BenignPositive). Confermato consistente per incidente. Target per classificazione. |
| 11 | ActionGrouped        | VARCHAR                      | 0.046%      | ALERT                 | (OrgId, AlertId)                           | Azione di remediation aggregata (solo subset etichettato). Utile per task “remediation prediction”. |
| 12 | ActionGranular       | VARCHAR                      | 0.046%      | EVIDENCE/ALERT         | (OrgId, AlertId)                           | Azione di remediation dettagliata. Molto sparsa; usare solo quando presente o per task secondario. |
| 13 | EntityType           | VARCHAR                      | 100%        | EVIDENCE              | (OrgId, AlertId, EntityType)               | Tipo di evidenza/entità della riga (User, Ip, CloudLogonRequest, …). Spiega perché esistono più righe per lo stesso AlertId. |
| 14 | EvidenceRole         | VARCHAR                      | 100%        | EVIDENCE              | (OrgId, AlertId, EvidenceRole)             | Ruolo dell’evidenza: Impacted (principale) vs Related (contesto). Utile per pesare/filtrare evidenze. |
| 15 | DeviceId             | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, DeviceId)                 | ID dispositivo (anon.). Alta cardinalità; utile come conteggio/presenza (n_device per alert/incidente). |
| 16 | Sha256               | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, Sha256)                   | Hash file (anon.). Alta cardinalità; utile in aggregazione (n_hash, presenza hash). |
| 17 | IpAddress            | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, IpAddress)                | IP (anon.). Può variare tra evidenze dello stesso alert; utile per n_ips/n_countries e pattern geo. |
| 18 | Url                  | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, Url)                      | URL (anon.). Utile come indicatore/contatore (n_url) più che come valore diretto. |
| 19 | AccountSid           | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, AccountSid)               | Identificatore account (SID anon.). Altissima cardinalità; usare in aggregazione (n_accounts). |
| 20 | AccountUpn           | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, AccountUpn)               | UPN account (anon.). Utile per raggruppare attività su identità (conteggi). |
| 21 | AccountObjectId      | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, AccountObjectId)          | ObjectId (Azure AD) anon. Utile per correlare eventi cloud su identità. |
| 22 | AccountName          | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, AccountName)              | Nome account codificato (anon.). Feature categoriale (da aggregare). |
| 23 | DeviceName           | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, DeviceName)               | Nome device codificato (anon.). Utile per correlazione tra eventi dello stesso host. |
| 24 | NetworkMessageId     | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, NetworkMessageId)         | ID messaggio rete/email correlato. Utile per linkare eventi (campagne, thread, ecc.). |
| 25 | EmailClusterId       | DOUBLE                       | 1%          | EVIDENCE              | (OrgId, AlertId, EmailClusterId)           | Cluster email sospette (solo subset). Utile per phishing/email analytics; molto mancante. |
| 26 | RegistryKey          | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, RegistryKey)              | Chiave di registro (anon.). Indica attività su host (tipico Windows); utile per pattern. |
| 27 | RegistryValueName    | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, RegistryValueName)        | Nome del valore nel registro (anon.). Utile per correlazioni/pattern di persistenza. |
| 28 | RegistryValueData    | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, RegistryValueData)        | Contenuto del valore di registro (anon.). Alta cardinalità; meglio usarlo in forma aggregata. |
| 29 | ApplicationId        | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, ApplicationId)            | ID applicazione (anon.). Utile per contesto applicativo dell’evento. |
| 30 | ApplicationName      | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, ApplicationName)          | Nome applicazione codificato (anon.). Feature categoriale (da aggregare). |
| 31 | OAuthApplicationId   | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, OAuthApplicationId)       | ID OAuth app (contesto cloud). Utile per accessi cloud e correlazioni su app. |
| 32 | ThreatFamily         | VARCHAR                      | 0.75%       | ALERT/EVIDENCE         | (OrgId, AlertId)                           | Famiglia minaccia (trojan/ransomware…). Molto sparsa; utile quando presente per interpretazione/feature. |
| 33 | FileName             | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, FileName)                 | Nome file codificato (anon.). Utile per pattern su file (da aggregare). |
| 34 | FolderPath           | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, FolderPath)               | Percorso file codificato (anon.). Utile per pattern di esecuzione/persistenza (da aggregare). |
| 35 | ResourceIdName       | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, ResourceIdName)           | Identificatore risorsa cloud (anon.). Utile per correlare eventi su risorse cloud. |
| 36 | ResourceType         | VARCHAR                      | 0.07%       | EVIDENCE              | (OrgId, AlertId, ResourceType)             | Tipo risorsa cloud. Quasi sempre mancante; utile solo nel subset cloud. |
| 37 | Roles                | VARCHAR                      | 2.6%        | EVIDENCE              | (OrgId, AlertId, Roles)                    | Ruoli associati (admin/user…). Sparso ma informativo (privilegi/rischio). |
| 38 | OSFamily             | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, OSFamily)                 | Famiglia OS codificata (Windows/Linux/…). Utile per segmentazione e pattern. |
| 39 | OSVersion            | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, OSVersion)                | Versione OS codificata. Utile come feature categoriale/di compatibilità. |
| 40 | AntispamDirection    | VARCHAR                      | 1.8%        | EVIDENCE              | (OrgId, AlertId, AntispamDirection)        | Direzione/contesto antispam (subset email). Utile quando presente. |
| 41 | SuspicionLevel       | VARCHAR                      | 15%         | ALERT/EVIDENCE         | (OrgId, AlertId)                           | Livello di sospetto/“confidence”. Sparso, ma utile come feature quando presente. |
| 42 | LastVerdict          | VARCHAR                      | 24%         | ALERT/EVIDENCE         | (OrgId, AlertId)                           | Verdettto finale (Blocked/Suspicious/Benign…). Utile per analisi errori e feature quando presente. |
| 43 | CountryCode          | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, CountryCode)              | Codice paese (anon.). Può cambiare tra evidenze dello stesso alert; utile per n_countries e pattern geo. |
| 44 | State                | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, State)                    | Stato/regione (anon.). Utile in aggregazione geografica. |
| 45 | City                 | BIGINT                       | 100%        | EVIDENCE              | (OrgId, AlertId, City)                     | Città (anon.). Utile in aggregazione (n_cities), può variare tra evidenze dello stesso alert. |
