# Nova Chatbot Anwendung

## Inhalt
1. [Voraussetzungen](#voraussetzungen)
2. [Installation](#installation)
3. [Anwendung starten](#anwendung-starten)
4. [Use Case anpassen](#use-case-anpassen)
5. [Projektstruktur](#projektstruktur)

---

## Voraussetzungen
Bevor du mit der Anwendung loslegst, stelle sicher, dass du Folgendes installiert und eingerichtet hast:

- **Python 3.8+**
- Einen Azure-Account mit:
  - Azure OpenAI Service (API-Schlüssel und Endpunkt)
  - Azure Cosmos DB (Endpoint, Key, Database und Container)
- Zugriff auf eine SQL-Datenbank (optional, je nach Agent-Konfiguration)
- Ein `.env`-File im Projekt-Root mit den folgenden Einträgen:

  ```dotenv
# OpenAI (Azure OpenAI Service)
OPENAI_API_VERSION="2024-08-01-preview"
OPENAI_API_KEY=<dein_openai_api_key>
OPENAI_ENDPOINT=<dein_openai_endpoint>
OPENAI_DEPLOYMENT_NAME_4o="gpt-4o"
OPENAI_DEPLOYMENT_NAME_35-turbo="gpt-35-turbo"
OPENAI_DEPLOYMENT_NAME_4omini="gpt-4o-mini"

# Cosmos DB
COSMOS_ENDPOINT=<dein_cosmos_endpoint>
COSMOS_KEY=<dein_cosmos_key>

# Embeddings (Ada)
ADA_ENDPOINT=<dein_embedding_endpoint>
ADA_EMBEDDING_KEY=<dein_embedding_schluessel>

# SQL-Datenbank (für relationalen Agenten, optional)
USERNAME_RELDB=<db_username>
PASSWORD_RELDB=<db_passwort>

# Weitere Einstellungen
maxTokens=1000

# (Optional) Azure SDK / CLI
AZURE_API_KEY=<dein_azure_api_key>
AZURE_ENDPOINT=<dein_azure_endpoint>

  Zu finden sind diese Keys im Microsoft Azure Workspace von Ceteris AG. Ebenso sind die Umgebungsvariablen in der Dokumentation enthalten und erklärt.
  ```

> **Hinweis:** Speichere das `.env`-File im Projekt-Root, damit `dotenv` es automatisch lädt.

---

## Installation
1. Repository klonen:
   ```bash
   git clone <dein-azure-repo-url> nova-chatbot
   cd nova-chatbot
   ```

2. Virtuelle Umgebung erstellen (empfohlen):
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Abhängigkeiten installieren:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Sicherstellen, dass dein `.env` korrekt gefüllt ist (siehe [Voraussetzungen](#voraussetzungen)).

---

## Anwendung starten
Nach der Installation kannst du den Flask-Server starten:

0. SQL-Server über VM starten:
   ```bash
   Um den agent_database.py ausführen zu können, muss die SQL-Datenbank manuell gestartet werden.
   ```

1. Umgebungsvariable für Flask setzen (optional):
   ```bash
   export FLASK_APP=app.py
   export FLASK_ENV=development   # für Debug-Modus
   ```

2. Server starten:
   ```bash
   python app.py

   oder direkt über app.py starten.
   ```

3. Öffne deinen Browser unter `http://localhost:5000`.

> Standard-Frontend liegt in `Interface/frontend/index.html`; Static-Assets in `Interface/static/`.

---

## Use Case anpassen
Die Standard-Implementierung ist als Bose-Support-Bot konfiguriert. Um den Chatbot für einen anderen Anwendungsfall (z. B. FAQ-Bot, Verkaufsberater) anzupassen, führe folgende Schritte aus:

1. **System-Prompts ändern**
   - Öffne `orchestrator.py` und passe `decision_system_content` sowie `process_agent_prompts` an dein neues Domänenwissen an.
   - In `agent_general.py`, `agent_vector.py` und `agent_database.py` kannst du in den `system_content`-Strings den Beschreibungstext entsprechend ändern.

2. **Routing-Logik prüfen**
   - Im `orchestrator.py` (`routing_node`) werden Anfragen den Agenten `general`, `vector` oder `database` zugewiesen. Passe bei Bedarf die Kriterien im Prompt oder die Schwellen für `confidence` an.

3. **Frontend-Texte anpassen**
   - In `Interface/frontend/index.html` kannst du Begrüßungstexte und UI-Beschriftungen für dein neues Szenario editieren.
   - Style-Anpassungen in `Interface/static/styles.css` sind ebenfalls möglich.

4. **Environment-Variablen erweitern**
   - Falls dein Use Case zusätzliche API-Schlüssel benötigt, erweitere die `.env`-Datei und passe `app.py` bzw. `load_dotenv()`-Aufrufe entsprechend an.

5. **Neustart & Test**
   - Nach Anpassungen den Server neu starten und in der Weboberfläche deine neuen Prompts und Workflows testen.

---

## Projektstruktur
```
Die Projektstruktur ist wie folgt aufgebaut: 

SOLUTION.HTW_RAG/
├── .gitignore
├── .env                        # Environment‑Variablen (nicht versionieren)
├── config.yaml                 # Zusätzliche Konfiguration (optional)
├── README.md
├── requirements.txt
│
├── Interface/                  # Web‑Frontend
│   ├── frontend/               # Single‑Page‑App
│   │   └── index.html
│   └── static/                 # Statische Assets
│       ├── Nova.png
│       └── styles.css
│
├── orchestration/              # Agenten‑Orchestrierung
│   └── completeStorage/        # Kern‑Module
│       ├── agent_database.py
│       ├── agent_general.py
│       ├── agent_vector.py
│       └── orchestrator.py
│
├── textprocessing/             # PDF‑Chunking & Vektor‑Speicher
│   ├── chunker.py
│   ├── ChunkerPSQL.py
│   ├── app.py                  # Hauptskript für Text‑Verarbeitung
│   ├── pdf_chunked/            # Zwischen­ge­speicherte PDF‑Chunks
│   ├── pdf_manuals/
│   ├── test_pdfs/
│   └── VectorStore/            # FAISS‑Index & PSQL‑Backend
│
├── relDB_Agent/                # (Optional) Relationale‑DB‑Agenten
│   └── …
│
└── backend/                    # Tests & Hilfs­skripte
    └── chat_history_tests.py   # Unit‑Tests

```

---

# Hinweis: Eine ausführliche Dokumentation finden Sie in Ceteris. Diese wurde mit der Projektgruppe der HTW im Modul "Projekt Business Intelligence" erstellt.

Autoren: 

- Lennox Lingk
- Leander Piepenbring
- Tobias Lindhorst
- Maximilian Berthold