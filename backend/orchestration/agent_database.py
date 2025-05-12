from langchain_community.utilities import SQLDatabase  # Importiert Utility-Klasse zum Verbinden mit SQL-Datenbanken
from typing_extensions import TypedDict  # Ermöglicht strenge Typdefinition für Dictionaries
from langchain_openai import AzureChatOpenAI  # Importiert Azure-basierten LLM-Wrapper
from langchain.prompts import PromptTemplate  # Ermöglicht das Definieren von LLM-Prompts mit Platzhaltern
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool  # Tool zum Ausführen von SQL-Abfragen
from langgraph.graph import START, StateGraph  # Für das Erstellen und Steuern von Graph-basierten Flows
from typing_extensions import Annotated  # Für annotierte Typen (nicht verwendet, aber importiert)
from dotenv import load_dotenv  # Lädt Umgebungsvariablen aus einer .env-Datei

import yaml  # Zur Verarbeitung von YAML-Dateien (aktuell nicht genutzt)
import getpass  # Zum sicheren Abfragen des Benutzernamens/PW (nicht genutzt)
import os  # Für Zugriff auf Umgebungsvariablen
import requests  # HTTP-Anfragen (in diesem Code nicht genutzt)
import json  # JSON-Verarbeitung für Ein-/Ausgabe
import re  # Reguläre Ausdrücke für String-Bereinigung

# Umgebungsvariablen laden
load_dotenv()

# Anmeldeinformationen aus Umgebungsvariablen beziehen
USERNAME = os.environ.get("USERNAME_RELDB")
PASSWORD = os.environ.get("PASSWORD_RELDB")
    
# Verbindungsdaten für die SQL-Datenbank konfigurieren
HOST = "demo-htw.database.windows.net"
PORT = "1433"
DATABASE = "adventureworks"
DRIVER = "ODBC+Driver+18+for+SQL+Server"

# Erstellen der Verbindungs-URI im SQLAlchemy-konformen Format
connection_uri = (
    f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    f"?driver={DRIVER}"
)

# Initialisieren der SQLDatabase-Verbindung
db = SQLDatabase.from_uri(connection_uri)

## Typdefinitionen für den Zustand und SQL-Ausgabe
# Klasse für den Status aller Schritte im Graphen
class State(TypedDict, total=False):
    question: str
    email: str
    global_summary: str
    global_buffer: str
    query_feedback: str
    query: str
    result: str
    answer: str

# Struktur der zurückzugebenden SQL-Abfrage
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: str  # "Syntaktisch korrekte SQL-Abfrage"

# Azure Chat LLM-Instanz zur Generierung von SQL-Queries
llm_prompt = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME_4o"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)

# Azure Chat LLM-Instanz für Validierung und Nachbearbeitung
llm_process = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME_4omini"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
)

# Prompt-Vorlage für das Generieren einer SQL-Abfrage
sql_query_prompt = """
Given an input question, create a syntactically correct {dialect} query to run to help find the answer. This dialect supports standard SQL syntax such as SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, and DROP. Functions like paging with OFFSET and FETCH NEXT are also available. However, there are limitations: USE database_name and file system functions are not supported, nor are FileStream, CLR, SQL Agent, and Linked Server. Top in Combination with Offset is not allowed in the same query.
You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for the few relevant columns given the question.
Never query to create or delete any tables or databases.
Pay attention to use only the column names that you can see in the schema description. 
Be careful not to query for columns that do not exist. Also, pay attention to which column is in which table.

This is the database structure you must only use:
{database_structure}

Use the following E_mail filter if available: {email_clause}

Das ist eine Zusammenfassung des bisherigen Chats:
{summary_clause}

Dies sind die letzten 3 Nachrichten. Falls der Kontext relevant ist, beachte diese mehr als die Zusammenfassung.
{buffer_clause}

This is the feedback from previous attempts:
{feedback_clause}

Question: {input}

Please provide the SQL query in the following JSON format:
{{
    "query": "YOUR_SQL_QUERY_HERE"
}}
"""
# Datenbankstruktur
database_structure = """
SalesLT.Customer(CustomerID, NameStyle, FirstName, MiddleName, LastName, Suffix, CompanyName, SalesPerson, EmailAddress, Phone, PasswordHash, PasswordSalt, rowguid, ModifiedDate), SalesLT.Address(AddressID, AddressLine1, AddressLine2, City, StateProvince, CountryRegion, PostalCode, rowguid, ModifiedDate), SalesLT.CustomerAddress(CustomerID, AddressID, AddressType, rowguid, ModifiedDate), SalesLT.Product(ProductID, Name, ProductNumber, StandardCost, ListPrice, ProductCategoryID, SellStartDate, SellEndDate, DiscontinuedDate, rowguid, ModifiedDate), SalesLT.SalesOrderDetail(SalesOrderID, SalesOrderDetailID, OrderQty, ProductID, UnitPrice, UnitPriceDiscount, LineTotal, rowguid, ModifiedDate), SalesLT.SalesOrderHeader(SalesOrderID, RevisionNumber, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag, SalesOrderNumber, PurchaseOrderNumber, AccountNumber, CustomerID, ShipToAddressID, BillToAddressID, ShipMethod, CreditCardApprovalCode, SubTotal, TaxAmt, Freight, TotalDue, Comment, rowguid, ModifiedDate)
"""

# Zusammensetzen des PromptTemplates mit allen benötigten Variablen
prompt = PromptTemplate(
    template=sql_query_prompt,
    input_variables=["dialect", "input", "email_clause", "summary_clause", "buffer_clause", "feedback_clause", "database_structure"]
)

def clean_json_response(response_str: str) -> str:
    # Entfernt Markdown-Codeblöcke und whitespace, extrahiert JSON-Block
    cleaned = re.sub(r"^```(json)?", "", response_str).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end+1]
    return cleaned

# Validierung des Outputs durch LLM
def validate_output(state: State, generated_output: str) -> dict:
    # Prompt zur Überprüfung, ob Antwort korrekt und vollständig ist
    output_validation_prompt = f"""
Bitte überprüfe das folgende SQL-Ergebnis und die generierte Antwort basierend auf der Frage: "{state["question"]}".
SQL-Ergebnis und Antwort:
{generated_output}

Wenn der Output die Frage beantwortet, antworte mit "ok". 
Sollte kein Output vorhanden sein wenn keine E-Mail angegeben ist, antworte ebenfalls mit "ok".
Bitte antworte ausschließlich im folgenden JSON-Format:
{{
    "rationale": "Begründung in 1-2 Sätzen, warum die Abfrage korrekt ist oder nicht.",
    "decision": "ok|not ok",
    "confidence": "eine Zahl zwischen 0 und 1"
}}
"""
    response = llm_process.invoke(output_validation_prompt)
    try:
        cleaned_response = clean_json_response(response.content)
        if not cleaned_response.startswith("{"):
            raise ValueError("Antwort beginnt nicht mit '{'")
        output_feedback = json.loads(cleaned_response)
    except Exception as e:
        print("Fehler beim Parsen des Output-Feedbacks:", e)
        output_feedback = {"decision": "not ok", "rationale": "Kein gültiges Output-Feedback erhalten.", "confidence": 0.0}
    return output_feedback

# Generiert eine SQL-Abfrage anhand des aktuellen Zustands
def write_query(state: State):
    email = state.get("email", "").strip()
    # E-Mail-Filter für anonym oder konkret
    if email and email.lower() != "anonymous":
        email_clause = f"WHERE EmailAddress = '{email}'"
    else:
        email_clause = "-- Zugriff verweigert: anonymer Nutzer"
    
    summary_history = state.get("global_summary", "").strip()
    buffer_history = state.get("global_buffer", "").strip()
    
    feedback = state.get("query_feedback", "").strip()
    feedback_clause = ""
    if feedback:
        feedback_clause = f"Feedback from previous attempts: {feedback}"
    
    filled_prompt = prompt.format(
        dialect=db.dialect,
        input=state["question"],
        email_clause=email_clause,
        summary_clause=summary_history,
        buffer_clause=buffer_history,
        feedback_clause=feedback_clause,
        database_structure=database_structure
    )
    
    try:
        structured_llm = llm_prompt.with_structured_output(QueryOutput)
        result = structured_llm.invoke(filled_prompt)
        return {"query": result["query"]}
    except Exception as e:
        print("Fehler beim Generieren der SQL-Abfrage:", e)
        return None

# Validiert die generierte SQL-Abfrage auf Richtigkeit
def validate_query(state: State, generated_query: str) -> dict:
    validation_prompt = f"""
Bitte überprüfe die folgende SQL-Abfrage basierend auf der Frage: "{state["question"]}".
Achte dabei auf:
- Wird die Tabelle Customer, CustomerAddress, SalesOrderDetail oder SalesOrderHeader verwendet, muss eine WHERE-Klausel mit der E-Mail Adresse des Kunden hinzugefügt sein.

Die Datenbankstruktur ist wie folgt:
{database_structure}

SQL-Abfrage: {generated_query}

Bitte antworte ausschließlich im folgenden JSON-Format:
{{
    "rationale": "Begründung in 1-2 Sätzen, warum die Abfrage korrekt ist oder nicht.",
    "decision": "ok|not ok",
    "confidence": "eine Zahl zwischen 0 und 1"
}}
"""
    response = llm_process.invoke(validation_prompt)
    print("Validierungsoutput:", response.content)
    
    try:
        cleaned_response = clean_json_response(response.content)
        feedback_json = json.loads(cleaned_response)
    except Exception as e:
        print("Fehler beim Parsen des JSON-Feedbacks:", e)
        feedback_json = {"decision": "not ok", "rationale": "", "confidence": 0.0}
    return feedback_json

# Hauptlogik: Chain-of-Thought mit bis zu 3 Versuchen
def write_query_with_chain_of_thought(state: State):
    max_attempts = 3
    last_query = ""
    for attempt in range(max_attempts):
        print(f"Versuch {attempt+1} der Query-Generierung")
        generated = write_query(state)
        if not generated:
            print("Fehler beim Generieren der Query.")
            continue
        query = generated["query"]
        last_query = query
        feedback = validate_query(state, query)
        if feedback.get("decision", "").lower() == "ok":
            return {"query": query}
        else:
            current_feedback = state.get("query_feedback", "")
            state["query_feedback"] = current_feedback + f"\nVersuch {attempt+1}: {feedback.get('rationale', '')}"
    return {"query": last_query}

# Ausführen der finalen SQL-Abfrage gegen die DB
def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    print(state["query"])
    print(execute_query_tool.invoke(state["query"]))
    return {"result": execute_query_tool.invoke(state["query"])}

# Generierung der finalen Antwort basierend auf Query-Ergebnis
def generate_answer(state: State):
    """
    Bitte antworte ausschließlich im folgenden JSON-Format, ohne zusätzlichen Text:
    {
        "result": "Deine Antwort als Fließtext.",
        "data": [Das SQL Ergebnis.]
    }
    """
    prompt_text = (
        "Beantworte die Frage des Benutzers basierend auf der SQL-Abfrage und dem SQL-Ergebnis.\n"
        "Bitte antworte ausschließlich im folgenden JSON-Format, ohne zusätzlichen Text:\n"
        "Verwende deutsche Spaltenbezeichnungen."
        "{\n"
        '  "result": "Deine Antwort als Fließtext. Gebe nur hier keine Listen mit mehr als 3 Produkten aus.",\n'
        '  "data": [Das vollständige SQL Ergebnis mit allen Einträgen. ]\n'
        "}\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}\n"
    )
    response = llm_process.invoke(prompt_text)
    cleaned_response = clean_json_response(response.content)
    return {"answer": cleaned_response}

# Aufbau des Graphen für den gesamten Workflow
graph_builder = StateGraph(State).add_sequence(
   [write_query_with_chain_of_thought, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query_with_chain_of_thought")
graph = graph_builder.compile()

# Funktion zur Handhabung einer Benutzerabfrage
def handle_database_query(user_message: str, email: str = "", global_summary: str = "", global_buffer: str = "") -> str:
    max_attempts = 3
    attempt = 0
    final_answer = ""
    while attempt < max_attempts:
        attempt += 1
        initial_state = {
            "question": user_message,
            "email": email,
            "global_summary": global_summary,
            "global_buffer": global_buffer,
            "query_feedback": "",
            "query": "",
            "result": "",
            "answer": ""
        }
        try:
            final_state = {}
            for step in graph.stream(initial_state, stream_mode="updates"):
                final_state = step

            # Extrahiere die Antwort aus dem finalen State
            if "answer" in final_state and final_state["answer"]:
                final_answer = final_state["answer"]
            elif "postprocess_node" in final_state and final_state["postprocess_node"].get("processed_output"):
                final_answer = final_state["postprocess_node"]["processed_output"]
            elif ("generate_answer" in final_state and 
                  isinstance(final_state["generate_answer"], dict) and 
                  "answer" in final_state["generate_answer"]):
                final_answer = final_state["generate_answer"]["answer"]
            else:
                final_answer = ""
            
            print(final_answer)
            
            # Validierung des Outputs mithilfe des LLM
            output_feedback = validate_output(initial_state, final_answer)
            print(f"Output-Validierungsfeedback: {output_feedback}")
            if output_feedback.get("decision", "").lower() == "ok":
                return final_answer
            else:
                print(f"Output-Validierung fehlgeschlagen bei Versuch {attempt}. Neue Query wird generiert.")
        except Exception as e:
            print(f"Fehler in handle_query_database bei Versuch {attempt}: {e}")
    return "Entschuldigung, es konnte keine gültige Antwort generiert werden. agent_database"

