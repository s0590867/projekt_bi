import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from typing_extensions import TypedDict

# Umgebungsvariablen laden
load_dotenv()

# Initialisiere den allgemeinen Agenten.
# Achte darauf, dass in Deiner .env-Datei ein passender Deployment-Name (z.B. OPENAI_DEPLOYMENT_NAME_GENERAL) gesetzt ist.
general_agent = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_ENDPOINT"],
    azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME_4o"],
    openai_api_version=os.environ["OPENAI_API_VERSION"]
)

# System Prompt inklusive Hinweis auf den Gesprächsverlauf (Chain of Thought)
general_system_content = (
    "Du bist ein allgemeiner Assistent, der auf eine Vielzahl von Fragen antworten kann. "
    "Du erhältst den bisherigen Konversationsverlauf, um den Kontext zu verstehen, und du gibst dem Benutzer klare, präzise und hilfreiche Antworten. "
    "Berücksichtige den bisherigen Verlauf, ohne sensible Daten zu wiederholen."
)

# Definiere einen Prompt, der den Systeminhalt, den Gesprächsverlauf (history) und die aktuelle Frage (input) integriert.
general_prompt_template = """
{system_content}

Das ist eine Zusammenfassung des bisherigen Chats:
{summary_clause}

Dies sind die letzten 3 Nachrichten. Falls der Kontext relevant ist, beachte diese mehr als die Zusammenfassung.
{buffer_clause}

Benutzerfrage: {input}

Bitte antworte präzise und hilfreich in Deutsch.
"""

general_prompt = PromptTemplate(
    template=general_prompt_template,
    input_variables=["system_content", "input", "summary_clause", "buffer_clause"]
)

class GeneralState(TypedDict, total=False):
    question: str
    global_summary: str
    global_buffer: str
    answer: str

def general_node(state: GeneralState) -> GeneralState:
    summary_clause = ""
    if state.get("global_summary", "").strip():
        summary_clause = state["global_summary"]
    buffer_clause = ""
    if state.get("global_buffer", "").strip():
        buffer_clause = state["global_buffer"]
    
    filled_prompt = general_prompt.format(
        system_content=general_system_content,
        input=state["question"],
        summary_clause=summary_clause,
        buffer_clause=buffer_clause
    )
    
    response = general_agent.invoke([
        SystemMessage(content=filled_prompt),
        HumanMessage(content=state["question"])
    ])
    state["answer"] = response.content.strip()
    return state

graph_general = StateGraph(GeneralState).add_sequence([general_node])
graph_general.add_edge(START, "general_node")
graph_general = graph_general.compile()

def handle_general_query(user_message: str, global_summary: str = "", global_buffer: str = "") -> str:
    """
    Verarbeitet eine allgemeine Benutzeranfrage unter Einbeziehung des bisherigen Gesprächsverlaufs.
    
    Args:
        user_message (str): Die aktuelle Frage des Benutzers.
        global_summary (str): Der bisherige Gesprächsverlauf zusammengefasst.
        global_buffer (str): Die letzten 3 Nachrichten.
        
    Returns:
        str: Die generierte Antwort des allgemeinen Agenten.
    """
    initial_state = {
        "question": user_message,
        "global_summary": global_summary,
        "global_buffer": global_buffer,
        "answer": ""
    }
    final_state = {}
    for step in graph_general.stream(initial_state, stream_mode="updates"):
        final_state = step
    return final_state.get("answer", "Entschuldigung, ich konnte keine Antwort generieren. agent_general")
