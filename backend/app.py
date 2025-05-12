"""
Nova Chatbot – Backend (app.py)
================================

1.  Standard‑ & Drittanbieter‑Importe
2.  Konfiguration & Logging
3.  Cosmos DB (Client, Datenbank, Container)
4.  Large‑Language‑Model (Azure OpenAI) & Memory‑Store
5.  Flask‑Anwendung
6.  Routen / Endpoints
    6.1  GET  /                    – index
    6.2  POST /start-session       – start_session
    6.3  POST /chat                – chat
    6.4  POST /end-session         – end_session
    6.5  GET  /get-sessions        – get_sessions
    6.6  GET  /get-session/<id>    – get_session
    6.7  POST /select-session/<id> – select_session
7.  main‑Guard (Entwicklungs‑Server)
"""

# ==========================================================
# 1) Standard- und Drittanbieter‑Importe
# ==========================================================
import os
import uuid
import datetime
import logging
import traceback
import json
import openai

from flask import Flask, render_template, request, jsonify, session
from azure.cosmos import CosmosClient, PartitionKey
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_openai import AzureChatOpenAI

from orchestration.orchestrator import dispatch  # Zentrale Entscheidungslogik

# ==========================================================
# 2) Konfiguration & Logging
# ==========================================================
# Azure‑SDKs erzeugen sehr viele Log‑Einträge – wir reduzieren das auf ERROR
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)
logging.getLogger("azure").setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Cosmos DB – Verbindungsparameter aus Umgebungsvariablen (Fallbacks s.u.)
COSMOS_ENDPOINT  = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY       = os.environ.get("COSMOS_KEY")
COSMOS_DATABASE  = os.environ.get("COSMOS_DATABASE",  "ChatDB")
COSMOS_CONTAINER = os.environ.get("COSMOS_CONTAINER", "Chats")

# ==========================================================
# 3) Cosmos DB (Client, DB, Container)
# ==========================================================
client    = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database  = client.create_database_if_not_exists(id=COSMOS_DATABASE)
container = database.create_container_if_not_exists(
    id=COSMOS_CONTAINER,
    partition_key=PartitionKey(path="/email")
)

# ==========================================================
# 4) LLM & Conversation Memories
# ==========================================================
# summarizer_llm wird ausschließlich für ConversationSummaryBufferMemory
# verwendet. Deployment‑Namen & Version stammen aus ENV‑Variablen.
summarizer_llm = AzureChatOpenAI(
    azure_endpoint      = os.environ["OPENAI_ENDPOINT"],
    azure_deployment    = os.environ["OPENAI_DEPLOYMENT_NAME_4o"],
    openai_api_version  = os.environ["OPENAI_API_VERSION"],
    model_name          = "gpt-4"
)

# Pro Chat‑ID werden unterschiedliche Memory‑Objekte in diesem Dict geführt
memory_store: dict[str, dict] = {}

# ==========================================================
# 5) Flask‑App‑Initialisierung
# ==========================================================
# Ablagepfade für Frontend‑Dateien (relative zum Projekt‑Root)
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Interface/frontend'))
static_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Interface/static'))

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'dein_geheimer_schluessel'  # ⚠ In Produktion durch Secret Manager ersetzen

# ==========================================================
# 6) Routen / Endpoints
# ==========================================================

# ----------------------------------------------------------------------
# 6.1  Index – liefert das Frontend‑HTML (Single‑Page‑App)
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------------------------------------------------
# 6.2  /start-session – Neue Session anlegen / bestehende übernehmen
# ----------------------------------------------------------------------
@app.route('/start-session', methods=['POST'])
def start_session():
    new_email   = request.form.get('email') or "anonymous"
    old_email   = session.get('email', 'anonymous')
    old_chat_id = session.get('chat_id')  # kann None sein

    # ---------- Sonderfall ------------------------------------------------
    # Ein anonymer Chat wird nachträglich einer konkreten E‑Mail zugeordnet.
    if old_email == "anonymous" and new_email != "anonymous" and old_chat_id:
        try:
            old_doc = container.read_item(item=old_chat_id, partition_key=old_email)
        except Exception:
            old_doc = None

        new_chat_id = f"{new_email}-{uuid.uuid4().hex[:8]}"

        if old_doc:
            # Dokument kopieren unter neuer ID + Metadaten anpassen
            old_doc["id"]        = new_chat_id
            old_doc["email"]     = new_email
            old_doc["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            container.create_item(body=old_doc)
            container.delete_item(item=old_chat_id, partition_key=old_email)

        # Memory‑Eintrag umhängen, falls vorhanden
        if old_chat_id in memory_store:
            memory_store[new_chat_id] = memory_store.pop(old_chat_id)

        # Session‑Cookies aktualisieren
        session['email']   = new_email
        session['chat_id'] = new_chat_id
        return jsonify({'chat_id': new_chat_id})

    # ---------- Normalfall ------------------------------------------------
    # Neue Session (E‑Mail -> neue Session oder anonymous‐>anonymous Refresh)
    session_id = f"{new_email}-{uuid.uuid4().hex[:8]}"
    session['email']   = new_email
    session['chat_id'] = session_id

    # Conversation Memories anlegen
    memory_store[session_id] = {
        "global_buffer": ConversationBufferWindowMemory(memory_key="history", return_messages=False, k=3),
        "global_summary": ConversationSummaryBufferMemory(
            memory_key="history",
            return_messages=False,
            llm=summarizer_llm,
            max_token_limit=300
        )
    }
    return jsonify({'chat_id': session_id})

# ----------------------------------------------------------------------
# 6.3  /chat – Zentrale Chat‑Logik (Frage → Antwort)
# ----------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Payload immer als JSON verarbeiten (force=True: ignoriert Content‑Type)
        data = request.get_json(force=True)
        user_message_text = data.get('message', '')
        chat_id          = data.get('chat_id')

        if not user_message_text:
            return jsonify({'error': 'Nachricht fehlt'}), 400

        # -------------------------------------------------------------
        # Session‑Setup (falls Client noch keine chat_id besitzt)
        # -------------------------------------------------------------
        if not chat_id:
            email  = session.get('email') or "anonymous"
            chat_id = f"{email}-{uuid.uuid4().hex[:8]}"
            session['email']   = email
            session['chat_id'] = chat_id
            # Drei parallele Memories für unterschiedliche Agenten‑Wege
            memory_store[chat_id] = {
                "general":   ConversationBufferWindowMemory(k=10, memory_key="history", return_messages=False),
                "vector":    ConversationBufferWindowMemory(k=10, memory_key="history", return_messages=False),
                "database":  ConversationBufferWindowMemory(k=10, memory_key="history", return_messages=False)
            }
            chat_document = {
                "id":         chat_id,
                "email":      email,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "messages":   []
            }
            container.create_item(chat_document)
        else:
            # Chat‑ID vorhanden → Email aus Session übernehmen (fallback anonymous)
            email = session.get('email') or "anonymous"
            session['chat_id'] = chat_id

        # Memory‑Dict sicherstellen
        if chat_id not in memory_store:
            memory_store[chat_id] = {
                "global_buffer": ConversationBufferWindowMemory(k=3, memory_key="history", return_messages=False),
                "global_summary": ConversationSummaryBufferMemory(memory_key="history", return_messages=False, llm=summarizer_llm, max_token_limit=300)
            }

        # -------------------------------------------------------------
        # Anfrage an Orchestrator weiterleiten
        # -------------------------------------------------------------
        user_message = HumanMessage(content=user_message_text)
        user_message.email = email  # Zusatzattribut für Downstream‑Logik

        memory_dict = memory_store[chat_id]
        result     = dispatch(user_message, memory=memory_dict)
        answer     = result.get("answer")
        decision   = result.get("decision", "general")

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # -------------------------------------------------------------
        # Chat‑Verlauf in Cosmos persistieren
        # -------------------------------------------------------------
        try:
            chat_doc = container.read_item(item=chat_id, partition_key=email)
        except Exception:
            chat_doc = {
                "id":         chat_id,
                "email":      email,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "messages":   []
            }
            container.create_item(chat_doc)

        chat_doc["messages"].append({
            "sender":    "user",
            "content":   user_message_text,
            "timestamp": timestamp
        })
        chat_doc["messages"].append({
            "sender":    "bot",
            "content":   answer,
            "timestamp": timestamp,
            "agent":     decision
        })
        container.replace_item(item=chat_doc["id"], body=chat_doc)

        return jsonify({'response': answer})

    # ---------------------------------------------------------
    # Azure OpenAI – Content Filter / 400 BadRequest
    # ---------------------------------------------------------
    except openai.BadRequestError as e:
        logging.exception("OpenAI BadRequestError im /chat Endpoint:")
        error_str = str(e)
        short_msg = "Fehler beim Aufruf von Azure OpenAI (400)."
        try:
            splitted = error_str.split("Error code: 400 - ")
            if len(splitted) > 1:
                error_json = json.loads(splitted[1])
                short_msg  = error_json["error"]["message"]
                cf_result  = error_json["error"].get("innererror", {}).get("content_filter_result", {})
                filtered_cats = [f"{cat} (Severity: {det.get('severity','?')})" for cat, det in cf_result.items() if det.get("filtered")]
                if filtered_cats:
                    short_msg += "\nGefilterte Kategorien: " + ", ".join(filtered_cats)
        except Exception:
            pass
        error_message = (
            "Die Anfrage wurde von Azure OpenAI gefiltert. "
            f"{short_msg} "
            "Bitte passe deine Eingabe an oder kontaktiere den Support."
        ).replace("\n", " ")
        return jsonify({"error": error_message}), 400

    # ---------------------------------------------------------
    # Generischer Fehlerfang
    # ---------------------------------------------------------
    except Exception as e:
        logging.exception("Allgemeiner Fehler im /chat Endpoint:")
        error_message = (
            "Entschuldigung, es ist ein unbekannter Fehler aufgetreten. "
            "Bitte versuche es später erneut oder kontaktiere den Support."
        ).replace("\n", " ")
        return jsonify({"error": error_message}), 500

# ----------------------------------------------------------------------
# 6.4  /end-session – Server‑seitige Session zurücksetzen
# ----------------------------------------------------------------------
@app.route('/end-session', methods=['POST'])
def end_session():
    session.pop('conversation_history', None)
    session.pop('email', None)
    session.pop('chat_id', None)
    return jsonify({'message': 'Chat-Sitzung wurde beendet.'})

# ----------------------------------------------------------------------
# 6.5  /get-sessions – Liste früherer Chat‑Sitzungen
# ----------------------------------------------------------------------
@app.route('/get-sessions', methods=['GET'])
def get_sessions():
    email = session.get('email') or "anonymous"
    if email == "anonymous":
        return jsonify([])
    query = (
        "SELECT c.id, c.created_at "
        "FROM c WHERE c.email = @email "
        "AND ARRAY_LENGTH(c.messages) > 0 "
        "ORDER BY c.created_at DESC"
    )
    parameters = [{"name": "@email", "value": email}]
    items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    return jsonify(items)

# ----------------------------------------------------------------------
# 6.6  /get-session/<id> – Kompletten Verlauf einer Session abrufen
# ----------------------------------------------------------------------
@app.route('/get-session/<chat_id>', methods=['GET'])
def get_session(chat_id):
    email = session.get('email') or "anonymous"
    if email == "anonymous":
        return jsonify({"error": "Anonyme Nutzer können keine Sitzungen abrufen"}), 403
    try:
        item = container.read_item(item=chat_id, partition_key=email)
        return jsonify(item)
    except Exception:
        return jsonify({"error": "Session nicht gefunden"}), 404

# ----------------------------------------------------------------------
# 6.7  /select-session/<id> – Auf vorhandene Session umschalten
# ----------------------------------------------------------------------
@app.route('/select-session/<chat_id>', methods=['POST'])
def select_session(chat_id):
    query = "SELECT * FROM c WHERE c.id = @chat_id"
    parameters = [{"name": "@chat_id", "value": chat_id}]
    items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    if not items:
        return jsonify({"error": "Chat nicht gefunden"}), 404

    chat_doc = items[0]
    session['chat_id'] = chat_id
    session['email']   = chat_doc.get("email", "anonymous")

    if chat_id not in memory_store:
        memory_store[chat_id] = {
            "global_buffer": ConversationBufferWindowMemory(k=3, memory_key="history", return_messages=False),
            "global_summary": ConversationSummaryBufferMemory(
                memory_key="history",
                return_messages=False,
                llm=summarizer_llm,
                max_token_limit=300
            )
        }
    mem_dict = memory_store[chat_id]

    # Vollständigen Verlauf ins Memory übertragen (für Re‑Prompting)
    messages = chat_doc["messages"]
    for i, m in enumerate(messages):
        if i < len(messages) - 3:
            mem_dict["global_summary"].save_context({"input": m["content"]}, {"output": ""})
        mem_dict["global_buffer"].save_context({"input": m["content"]}, {"output": ""})

    return jsonify({"message": "Session switched"}), 200

# ==========================================================
# 7) Entwicklungs‑Server starten (nur bei direktem Aufruf)
# ==========================================================
if __name__ == '__main__':
    app.run(debug=True)