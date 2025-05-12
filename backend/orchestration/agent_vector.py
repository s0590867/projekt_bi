import sys
import os
#bestimmt den aktuellen Ordner und das Projekt-Root-Verzeichnis
aktueller_ordner = os.path.dirname(__file__)
projekt_root = os.path.abspath(os.path.join(aktueller_ordner, '..', '..'))
if projekt_root not in sys.path:
    sys.path.insert(0, projekt_root)

#sys.path.append(os.path.abspath('../../backend/textprocessing'))
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from typing_extensions import TypedDict
from backend.textprocessing import chunker
from backend.textprocessing import ChunkerPSQL


class State(TypedDict, total=False):
    question: str
    global_buffer: str
    global_summary: str
    context: str
    answer: str

class VectorAgent():
    def __init__(self):
        # Initialisiert das AzureChatOpenAI-Modell mit Umgebungsvariablen
        self.model = AzureChatOpenAI(
            azure_endpoint=os.environ["OPENAI_ENDPOINT"],
            azure_deployment=os.environ["OPENAI_DEPLOYMENT_NAME_4o"],
            openai_api_version=os.environ["OPENAI_API_VERSION"]
        )
        # Definiert die Systemnachricht für das Modell
        self.system_content = (
            "Du bist ein Assistent der konkrete Fragen zu Bose-Produkten anhand von Text-Bausteinen beantwortet. "
            "Du erhältst den bisherigen Konversationsverlauf, um den Kontext zu verstehen, und du gibst dem Benutzer klare, präzise und hilfreiche Antworten. "
            "Berücksichtige den bisherigen Verlauf, ohne sensible Daten zu wiederholen."
        )
        # Initialisiert den Embedding-Client für die Textsuche
        self.embedding_client = chunker.AzureOpenAIEmbeddings(
            api_key=os.environ["ADA_EMBEDDING_KEY"],
            api_base=os.environ["ADA_ENDPOINT"],
            model= "text-embedding-ada-002"
        )
    
    # Extrahiert Kontext aus der Datenbank basierend auf der Nutzerfrage
    def retrieve_context(self, state: State) -> State:
        processor = ChunkerPSQL.PDFProcessor()
        #die Keywords müssen noch aus dem User-Input extrahiert werden und als Keyword-Filters an den Chunker übergeben werden

        # keywords holen mit chunkerpsql
        keywords = "{" + ",".join(processor.get_keywords_with_gpt(state["question"]).split(", ")) + "}"
        print("Die Keywords für die question sind:" + keywords)
        # Sucht relevante Text-Bausteine basierend auf Frage und Schlüsselwörtern
        results = processor.search_chunks(state["question"], keywords)
        state["context"] = "\n\n".join([res[1] for res in results])
        return state

    # Generiert eine Antwort basierend auf Kontext und Verlauf
    def generate_answer(self, state: State) -> State:
        summary_clause = ""
        if state.get("global_summary", "").strip():
            summary_clause = f"Globale Zusammenfassung:\n{state['global_summary']}\n\n"
        buffer_clause = ""
        if state.get("global_buffer", "").strip():
            buffer_clause = f"Globale Buffer:\n{state['global_buffer']}\n\n"
        
        prompt = (
            f"{self.system_content}\n\n"
            f"Kontext:\n{state.get('context', '')}\n\n"
            f"Das ist eine Zusammenfassung des bisherigen Chats: {summary_clause}"
            f"Dies sind die letzten 3 Nachrichten. Falls der Kontext relevant ist, beachte diese mehr als die Zusammenfassung.{buffer_clause}"
            f"Benutzeranfrage: {state['question']}"
        )
        response = self.model.invoke([
            SystemMessage(content=self.system_content),
            HumanMessage(content=prompt)
        ])
        state["answer"] = response.content
        return state
    
# Hauptfunktion zum Verarbeiten einer Nutzeranfrage
def handle_vector_query(user_message: str, global_summary: str = "", global_buffer: str = "") -> str:
    load_dotenv()
    agent = VectorAgent()
    initial_state: State = {
        "question": user_message,
        "global_summary": global_summary,
        "global_buffer": global_buffer,
        "context": "",
        "answer": ""
    }
    # Erstellt einen StateGraph für die Verarbeitung
    graph = (
        StateGraph(State)
        .add_sequence([agent.retrieve_context, agent.generate_answer])
    )
    graph.add_edge(START, "retrieve_context")
    graph = graph.compile()
    
    final_state = {}
    for step in graph.stream(initial_state, stream_mode="updates"):
        final_state = step
    
    if "answer" in final_state and final_state["answer"]:
        return final_state["answer"]
    else:
        return "Entschuldigung, ich konnte keine Antwort generieren. agent_vector"



if __name__ == "__main__":
    load_dotenv()
    agent = VectorAgent()
    print(agent.system_content)
    print(agent.get_response(HumanMessage(content="Wie verbinde ich die Bose S1 Pro mit meinem Fernseher?")))