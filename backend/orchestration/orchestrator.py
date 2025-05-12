import os
import logging
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
import json
import re

load_dotenv()

# Initialisierung des Decision Agents (Router)
decision_agent = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME_4omini"],  # z.B. "gpt-35-turbo"
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    model_name="gpt-4o-mini",  # offizielle Modellbezeichnung
    response_format={"type": "json_object"}  # liefert JSON
)

# System-Prompt für den Decision Agent
decision_system_content = """
Du bist ein Dispatcher, der entscheidet, welcher Agent eine gegebene Benutzeranfrage am besten beantworten kann.
Entscheide dich zwischen 'vector' für Fragen, die sich konkret auf Bose-Produkte-Anleitungen beziehen,
und 'database' für Fragen, die eine Datenbankabfrage erfordern, also Produkte und Informationen allgemein, Informationen über sich selber (Kunde anhand Mail) .

Wenn die Frage sich auf Produkte, Bestellungen, Preise oder ähnliche Themen bezieht, wähle 'database'.
'vector' bei Abfragen, wo Anleitungen gebraucht werden.

Betrachte auch die "History" bei Fragen die sich auf vorherige Antworten beziehen, um den Agent auszuwählen.

Bitte antworte in folgendem JSON-Format:
{
    "rationale": "Begründung in 1-2 Sätzen, warum du dich so entschieden hast.",
    "decision": "general|vector|database",
    "final_decision": "vector|database|general",
    "confidence": "eine Zahl zwischen 0 und 1, die angibt, wie sicher du in deiner Entscheidung bist"
}
"""

# ------------------------------------------------------------
# Process Agent: überarbeitet die Antwort des Fach-Agenten vor Ausgabe
# ------------------------------------------------------------
process_agent = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME_4omini"],
    openai_api_version=os.environ["OPENAI_API_VERSION"]
)

# Verschiedene Prompts für den Process Agent, um Formatierung, Sprache, Höflichkeit etc. zu steuern
process_agent_prompts = {
    "intro": (
        "Du bist Nova, ein Assistent für Bose-Produkte. "
        "Deine Aufgabe ist es, Kunden bei Fragen, Problemen oder Anfragen zu Bose-Produkten klar, präzise und hilfreich zu unterstützen. "
        "Du bekommst hierfür von einem LLM-Agenten eine generierte Antwort, die du überarbeiten und an den Kunden weitergeben sollst."
    ),
    "keep_content": (
        "WICHTIG: Behalte den Originaltext EXAKT bei. "
        "Du darfst nichts weglassen, nichts paraphrasieren und keine eigenen Sätze einfügen. "
        "Erlaube dir nur Folgendes:\n"
        "1. Formatiere den Text in HTML (z.B. Absätze, Überschriften, Listen).\n"
        "2. Wenn du Höflichkeitsfloskeln ergänzen willst, füge sie in einem NEUEN Absatz <p> ...</p> am Ende an.\n"
        "3. Ändere keinesfalls den Kerninhalt, die Reihenfolge oder die Formulierungen.\n"
        "4. Entferne nichts.\n"
    ),
    "memory_limit": (
        "Falls der Kunde nach sehr alten Nachrichten fragt, die nicht in den letzten k=10 Nachrichten enthalten sind, "
        "erkläre, dass du nur auf die letzten 10 Nachrichten Zugriff hast und bitte den Kunden um mehr Kontext."
    ),
    "guide": (
        "Falls der Kunde nach einer Anleitung fragt, teile diese in klare, verständliche Schritte auf. "
        "Beginne mit einem kurzen Einführungstext, der den Zweck der Anleitung erklärt."
    ),
    "guide_format": (
        "Erstelle Anleitungen im HTML-Format mit folgender Struktur:\n"
        "- Überschriften (<h2>), um Hauptabschnitte zu kennzeichnen.\n"
        "- Absätze (<p>), um Details klar zu erklären.\n"
        "- Listen (<ul>, <li>), um Schritte oder Punkte übersichtlich darzustellen.\n"
        "- Verwende Formatierungen wie <b>Fett</b> und <i>Kursiv</i>, um wichtige Begriffe hervorzuheben."
    ),
    "paragraph_structure": (
        "Formuliere Antworten immer so, dass:\n"
        "- Jeder Absatz eine klare Idee oder einen Punkt abdeckt.\n"
        "- Absätze nicht länger als 3-4 Sätze sind.\n"
        "- Unterpunkte für Aufzählungen genutzt werden, wenn mehrere Möglichkeiten dargestellt werden."
    ),
    "followup": (
        "Am Ende jeder Antwort frage, ob die Informationen hilfreich waren oder ob weitere Details benötigt werden. "
        "Verwende Emotes sparsam, um die Antworten freundlicher zu machen."
    ),
    "language": (
        "Antworte immer in Deutsch, es sei denn, der Kunde schreibt auf Englisch. "
        "Passe die Sprache an den Kunden an, falls erforderlich."
    ),
    "sensitive_data_policy": (
        "Du darfst sensible Daten nur auf Anfrage des Kunden ausgeben, wenn diese eindeutig dem aktuellen Kunden gehören.\n"
        "- Beispiele für sensible Daten: Name, Bestellmenge, persönliche Adressen, Bestellungen, Rechnungsdaten.\n"
        "- Sensible Daten anderer Kunden oder generelle Informationen zu anderen Nutzern dürfen NICHT ausgegeben werden.\n"
        "- Falls ein Kunde Informationen zu allgemeinen Produkten oder Diensten anfragt, gib diese ohne Bezug zu spezifischen Kunden aus.\n"
        "- Falls der Kunde nach sensiblen Daten fragt, stelle sicher, dass die Anfrage eindeutig auf die eigene Identität verweist."
    ),
    "agent_database": (
        "Wenn ein JSON in der Ausgabe vorliegt, wandle dieses in eine HTML Tabelle um. \n"
        "Kürze Nachkommastellen auf maximal 2 Stellen. \n"
        "Verwende auf keinen Fall die Begriffe SQL, Datenbank oder ähnliches."
        "Verwende je nach Kontext die passenden Begriffe wie 'Produkte', 'Bestellungen', 'Preise' oder 'Informationen'."
    )
}

system_content = "\n\n".join(process_agent_prompts.values())

def safe_json_parse(response_text):
    """Versucht, JSON zu parsen; bei Fehlern wird None zurückgegeben."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.error("Ungültige JSON-Antwort vom LLM: %s", response_text)
        return None  # Fallback

def decision_node(state: dict) -> dict:
    # Baut die Eingaben für den Decision Agent zusammen und wertet dessen JSON-Antwort aus
    text = state.get("user_message", "")
    messages = [
        SystemMessage(content=decision_system_content),
        HumanMessage(content=f"Frage: {text}")
    ]
    response = decision_agent.invoke(messages)
    raw_response = response.content.strip()
    logging.debug("Decision Agent Raw Response: %s", raw_response)

    data = safe_json_parse(raw_response)
    if not data:
        raise ValueError("Die Antwort des Decision Agents ist kein gültiges JSON.")

    # Sicherstellen, dass alle wichtigen Felder vorhanden sind
    required_fields = ["decision", "final_decision", "rationale", "confidence"]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(f"Fehlende JSON-Felder: {missing_fields}")

    # Extrahiere und validiere die Felder
    decision = data["decision"].lower()
    final_decision = data["final_decision"].lower()
    rationale = data["rationale"]
    confidence = float(data["confidence"])

    # Bei geringer Sicherheit immer auf 'general' zurückfallen
    if confidence < 0.5:
        final_decision = "general"

    # Speichern im State-Dictionary
    state["decision"] = final_decision
    state["rationale"] = rationale
    state["confidence"] = confidence

    logging.debug("Decision Agent hat '%s' gewählt. Begründung: %s, Confidence: %s",
                  state["decision"], rationale, confidence)
    return state

def routing_node(state: dict) -> dict:
    # Lädt Verlauf und Zusammenfassung aus dem Memory
    memory_dict = state.get("memory_dict", {})
    global_buffer = memory_dict.get("global_buffer")
    global_summary = memory_dict.get("global_summary")
    buffer_history = ""
    if global_buffer:
        buffer_history = global_buffer.load_memory_variables({}).get("history", "")
    summary_history = ""
    if global_summary:
        summary_history = global_summary.load_memory_variables({}).get("history", "")
    
    state["global_summary"] = summary_history
    state["global_buffer"] = buffer_history

    text = state["user_message"]
    rationale = state.get("rationale", "")
    if rationale:
        text = f"Rationale: {rationale}\n{text}"
    email = state.get("email", "")

    # Dispatch an den gewählten Agenten
    decision = state.get("decision", "general")
    if decision == "database":
        from orchestration.agent_database import handle_database_query
        agent_answer = handle_database_query(text, email, summary_history, buffer_history)
    elif decision == "vector":
        from orchestration.agent_vector import handle_vector_query
        agent_answer = handle_vector_query(text, summary_history, buffer_history)
    else:
        from orchestration.agent_general import handle_general_query
        agent_answer = handle_general_query(text, summary_history, buffer_history)

    state["agent_output"] = agent_answer

    # Speichert die Unterhaltung in beiden Memories
    if global_buffer:
        global_buffer.save_context({"input": text}, {"output": agent_answer})
    if global_summary:
        global_summary.save_context({"input": text}, {"output": agent_answer})

    return state

def postprocess_node(state: dict) -> dict:
    # Holt die rohe Agenten-Antwort und startet das Post-Processing
    raw_output = state.get("agent_output", "")
    logging.debug("Postprocess Node, roher Agenten-Output: %s", raw_output)

    # Wenn Datenbank-Antwort (JSON) vorliegt, Tabelle aufbauen und Text formatieren
    if state.get("decision") == "database":
        try:
            parsed_output = json.loads(raw_output)
            # JSON mit 'result' und optional 'data'
            if isinstance(parsed_output, dict) and "result" in parsed_output:
                result_text = parsed_output["result"]
                data = parsed_output.get("data", [])
                # Text durch LLM formatieren lassen
                messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=f"Originalantwort des Agenten (Textteil): {result_text}")
                ]
                processed_text = process_agent.invoke(messages).content.strip()
                # Tabellenfragmente entfernen, um Duplikate zu vermeiden
                processed_text = re.sub(r'<table.*?>.*?</table>', '', processed_text, flags=re.DOTALL).strip()

                # Wenn 'data' eine Liste von Dicts ist, erzeugen wir eine HTML-Tabelle
                if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
                    from tabulate import tabulate

                    def format_value(val):
                        if isinstance(val, float):
                            return f"{val:.2f}"
                        return val

                    formatted_rows = []
                    for row in data:
                        formatted_row = {k: format_value(v) for k, v in row.items()}
                        formatted_rows.append(formatted_row)

                    html_table = tabulate(formatted_rows, headers="keys", tablefmt="html")
                    # Ergebnis: Verarbeiteter Text plus die Tabelle (die Tabelle wird nur einmal angehängt)
                    state["processed_output"] = f"{processed_text}\n{html_table}"
                else:
                    state["processed_output"] = processed_text

            # Fallback: JSON-Liste von Dicts direkt als Tabelle
            elif isinstance(parsed_output, list) and parsed_output and all(isinstance(item, dict) for item in parsed_output):
                from tabulate import tabulate

                def format_value(val):
                    if isinstance(val, float):
                        return f"{val:.2f}"
                    return val

                formatted_rows = []
                for row in parsed_output:
                    formatted_row = {k: format_value(v) for k, v in row.items()}
                    formatted_rows.append(formatted_row)
                html_table = tabulate(formatted_rows, headers="keys", tablefmt="html")
                state["processed_output"] = html_table
            else:
                # Alles andere unverändert lassen
                state["processed_output"] = raw_output

        except Exception as json_err:
            logging.debug("JSON Parsing fehlschlug: %s. Verwende Standard Postprocessing.", json_err)
            try:
                # Wenn JSON-Pfad scheitert, ganze Antwort noch einmal formatieren
                messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=f"Originalantwort des Agenten: {raw_output}")
                ]
                response = process_agent.invoke(messages)
                processed = response.content.strip()
                state["processed_output"] = processed
            except Exception as e:
                logging.error("Fehler beim Post-Processing: %s", e)
                state["processed_output"] = f"Die folgende Antwort konnte nicht weiter überarbeitet werden: {raw_output}"
    else:
        raw_output = state.get("agent_output", "")
        logging.debug("Postprocess Node, roher Agenten-Output: %s", raw_output)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=f"Originalantwort des Agenten: {raw_output}")
    ]
    try:
        response = process_agent.invoke(messages)
        processed = response.content.strip()
        state["processed_output"] = processed
        logging.debug("Process Agent Antwort: %s", processed)
    except Exception as e:
        logging.error("Fehler beim Post-Processing: %s", e)
        state["processed_output"] = f"Die folgende Antwort konnte nicht weiter überarbeitet werden: {raw_output}"
    logging.debug("State nach postprocess_node: %s", state)
    return state

# Aufbau und Kompilierung des StateGraph-Orchestrators
graph_orchestrator = (
    StateGraph(dict)
    .add_sequence([decision_node, routing_node, postprocess_node])
)
graph_orchestrator.add_edge(START, "decision_node")
graph_orchestrator = graph_orchestrator.compile()

# --------------------------------------------------------------------
# Hilfsfunktion: Extrahiere aus einem Text alle Fragen und kategorisiere sie
# --------------------------------------------------------------------
def extract_questions(text: str) -> list:
    extraction_prompt = f"""Extrahiere aus dem folgenden Text alle Fragen.
        Achte darauf, dass 2 Fragen per verbunden sein können und trenne diese ebenfalls!
        Wenn die Frage sich auf Produkte, Bestellungen, Preise oder ähnliche Themen bezieht, wähle 'database'.
        Wähle 'vector' bei Abfragen, wo Anleitungen gebraucht werden.
        Nur wenn die vorherigen Bedingungen keinen Sinn ergeben nutze 'general'.
        Erstelle aus den Fragen zu 'database' und 'vector' jeweils eine gesamte Frage, die alle entsprechenden Fragen enthält.
        Gib die Fragen als JSON-Liste aus. Achte darauf, dass alle Fragen vollständig sind.

        Nutze immer dieses Format:
        {{
            "database": [],
            "vector": [],
            "general": []
        }}

        Behalte jeweils Kontextinformationen bei, beispielsweise Produktnamen oder Kundeninformationen.
        Text:
        {text}"""
    messages = [
        SystemMessage(content="Du bist ein Assistent, der Fragen extrahiert."),
        HumanMessage(content=extraction_prompt)
    ]
    response = decision_agent.invoke(messages)
    result_str = response.content.strip()
    logging.debug("Response from extraction prompt: %s", result_str)

    # Suche nach dem JSON-Anfang und bereinige Code-Fences
    json_start = result_str.find("{")
    if json_start != -1:
        result_str = result_str[json_start:]
    else:
        logging.error("Kein JSON-Teil in der Antwort gefunden.")
        return [text]

    if result_str.startswith("```"):
        result_str = result_str.replace("```json", "").replace("```", "").strip()
        logging.debug("Cleaned JSON string: %s", result_str)

    if not result_str:
        logging.error("Die Antwort des LLM ist leer. Fallback wird aktiviert.")
        return [text]

    try:
        parsed = json.loads(result_str)
        # Verschiedene JSON-Layouts unterstützen
        if isinstance(parsed, dict):
            if "questions" in parsed:
                questions = parsed["questions"]
            elif all(key in parsed for key in ["database", "vector", "general"]):
                questions = parsed["database"] + parsed["vector"] + parsed["general"]
            else:
                questions = []
        else:
            questions = parsed

        if isinstance(questions, list) and questions:
            logging.info("Extracted questions: %s", questions)
            return questions
        else:
            logging.info("Keine Fragen extrahiert. Ursprünglicher Text wird als Liste zurückgegeben.")
            return [text]
    except Exception as e:
        logging.error("Fehler beim Extrahieren der Fragen: %s", e)
        logging.debug("Ungültige JSON-Antwort: %s", result_str)
        return [text]

# --------------------------------------------------------------------
# Hilfsfunktion: Kombiniert mehrere Teilsantworten zu einer finalen Antwort
# --------------------------------------------------------------------
def combine_answers(answers: list) -> str:
    combined_text = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
    prompt = (
        "Kombiniere bitte die folgenden Antworten zu einer einzigen, kohärenten Antwort. "
        "Stelle sicher, dass die finale Antwort fließend, verständlich und konsistent ist:\n" + combined_text
    )
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=prompt)
    ]
    response = process_agent.invoke(messages)
    return response.content.strip()

# --------------------------------------------------------------------
# Haupt-Funktion: Übergibt die Nutzeranfrage durch den Workflow und liefert die Antwort
# --------------------------------------------------------------------
def dispatch(user_message, history="", memory=None) -> dict:
    # Lade Verlauf und Summary aus dem Memory-Objekt, falls vorhanden
    if memory and isinstance(memory, dict):
        global_buffer = memory.get("global_buffer")
        global_summary = memory.get("global_summary")
        current_global_summary = ""
        current_global_buffer = ""
        if global_buffer and global_summary:
            current_global_buffer = global_buffer.load_memory_variables({}).get("history", "")
            current_global_summary = global_summary.load_memory_variables({}).get("history", "")
    elif memory is not None:
        current_global_summary = memory.load_memory_variables({}).get("history", "")
        current_global_buffer = ""
    else:
        current_global_summary = ""
        current_global_buffer = history

    # Extrahiere reinen Text aus user_message-Objekt
    if hasattr(user_message, "content"):
        text = user_message.content
    else:
        text = str(user_message)
    email = getattr(user_message, "email", "")

    # Zerlege bei Bedarf in mehrere Fragen
    questions = extract_questions(text)

    all_answers = []
    # Für jede Frage den Orchestrator durchlaufen
    for question in questions:
        current_global_summary = ""
        if memory and "global_buffer" in memory and "global_summary" in memory:
            current_global_buffer = memory["global_buffer"].load_memory_variables({}).get("history", "")
            current_global_summary = memory["global_summary"].load_memory_variables({}).get("history", "")
        state = {
            "user_message": question,
            "email": email,
            "memory_dict": memory,
            "decision": "",
            "agent_output": "",
            "processed_output": ""
        }
        final_state = {}
        for step in graph_orchestrator.stream(state, stream_mode="updates"):
            final_state = step
        answer = final_state.get("processed_output", "")
        if not answer and "postprocess_node" in final_state:
            answer = final_state["postprocess_node"].get("processed_output", "")
        if not answer:
            answer = final_state.get("agent_output", "Entschuldigung, es konnte keine Antwort generiert werden.")
        all_answers.append(answer)

    # Einzeln oder kombiniert zurückgeben
    if len(all_answers) == 1:
        final_answer = all_answers[0]
        decision = state.get("decision")
    else:
        final_answer = combine_answers(all_answers)
        decision = "multiple"
    return {"answer": final_answer, "decision": decision}
