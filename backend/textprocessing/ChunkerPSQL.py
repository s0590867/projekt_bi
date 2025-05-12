# Bibliotheken und Imports

import os
import sys
import base64
import psycopg2
import nltk
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from openai import AzureOpenAI

# NLTK-Modell für Tokenisierung laden
nltk.download('punkt')

# Arbeitsverzeichnis auf Skriptverzeichnis setzen
os.chdir(sys.path[0])

# Lade Umgebungsvariablen aus .env
load_dotenv()

# Klasse zum Erstellen von Embeddings via Azure OpenAI
class AzureOpenAIEmbeddings:
    
    def __init__(self):
        # Lade API-Keys & Endpunkte aus .env
        self.api_key = os.getenv("ADA_EMBEDDING_KEY")
        self.api_base = os.getenv("ADA_ENDPOINT")
        self.model = "text-embedding-ada-002"
        self.client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_base, api_version="2024-08-01-preview")

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


# Hauptklasse zur Verarbeitung von PDF-Dateien
class PDFProcessor:

    def __init__(self):
        # Lade API-Keys & Endpunkte aus .env
        self.gptTurboEndpoint = os.getenv("OPENAI_ENDPOINT")
        self.gptTurboKey = os.getenv("OPENAI_API_KEY")
        self.max_tokens = int(os.getenv("maxTokens", 1000))

        # Azure Document Intelligence (für OCR)
        self.endpoint = os.getenv("AZURE_ENDPOINT")
        self.key = os.getenv("AZURE_API_KEY")
        self.client = DocumentIntelligenceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))

        # Azure OpenAI Embedding Client
        self.embedding_client = AzureOpenAIEmbeddings()

        # PostgreSQL-Datenbankkonfiguration aus .env
        self.db_config = {
            "host": "psql-htw-proj.postgres.database.azure.com",
            "port": "5432",
            "dbname": "test-database",
            "user": os.getenv("USERNAME_RELDB"),
            "password": os.getenv("PASSWORD_RELDB"),
            "sslmode": "require"
        }

        self.pdf_files = []

    # Text aus PDF extrahieren
    def extract_text(self, pdf_path):
        with open(pdf_path, "rb") as f:
            base64_encoded_pdf = base64.b64encode(f.read()).decode("utf-8")

        poller = self.client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=base64_encoded_pdf))
        result = poller.result()

        # Text aus allen Seiten zusammensetzen
        text = " ".join(line.content for page in result.pages for line in page.lines)
        return text

    # Chunking des extrahierten Textes inkl. Overlap-Handling
    def chunk_text(self, text, max_tokens, overlap=5):
        sentences = nltk.sent_tokenize(text) # Text in Sätze aufteilen
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:]  # Overlap übernehmen
                current_tokens = sum(len(word.split()) for word in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # GPT-gestützte Keyword-Generierung auf Basis einer festen Liste
    def get_keywords_with_gpt(self, chunk):

        #Vordefinierte Keyword-Liste
        keywordsList = ["Bose", "Acoustimass", "Loudspeaker",
                        "Subwoofer", "Surround", "Bluetooth", "Wi-Fi",
                        "DSP", "Amplifier", "Crossover", "Bass",
                        "Treble", "Mounting", "Wireless", "Impedance", "Dolby",
                        "EQ", "Frequency", "Signal", "Audio"
                        ]
        prompt = (
            f"Select 1 to 3 keywords from the following predefined list that best describe the given text:\n\n"
            f"Keyword List: {', '.join(keywordsList)}\n\n"
            f"Text: {chunk}\n\n"
            f"Return only the selected keywords as a comma-separated list, without any explanations or extra text."
        )

        client = AzureOpenAI(
            api_key=self.gptTurboKey,
            azure_endpoint=self.gptTurboEndpoint,
            api_version="2024-08-01-preview"
        )

        chat_completion = client.chat.completions.create(
            model="gpt-35-turbo", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.5
        )

        if chat_completion and chat_completion.choices:
            message = chat_completion.choices[0].message
            if message and message.content:
                return message.content.strip()
        return "No keywords generated"

    # Alle PDFs im Ordnerpfad erfassen
    def set_pdf_files(self, folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    self.pdf_files.append(os.path.join(root, file))

    # Einfügen der Daten in die PostgreSQL-Datenbank
    def insert_into_db(self, filename, chunk_number, chunk_text, keywords, embedding):
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            query = """
            INSERT INTO chunks (filename, chunk_number, chunk_text, keywords, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """

            # Keywords als Liste für PostgreSQL-Array vorbereiten
            formatted_keywords = keywords.split(", ") if isinstance(keywords, str) else keywords

            cursor.execute(query, (filename, chunk_number, chunk_text, formatted_keywords, embedding))

            conn.commit()
            cursor.close()
            conn.close()
            print(f"Inserted chunk {chunk_number} from {filename} into the database.")

        except Exception as e:
            print("Error inserting into the database:", e)

    # Vektorbasierte Suche in der Datenbank mit optionalem Keyword-Filter
    def search_chunks(self, query_text, keywords_filter=None, limit=3):
        
        query_embedding = self.embedding_client.embed_query(query_text)

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            if keywords_filter:
                # Wenn ein Keyword-Filter gesetzt ist, wird dieser in der WHERE-Klausel verwendet.
                query = """
                SELECT id, chunk_text, keywords
                FROM chunks
                WHERE keywords && %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """
                params = (keywords_filter, query_embedding, limit)
            else:
                # Ohne Keyword-Filter
                query = """
                SELECT id, chunk_text, keywords
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """
                params = (query_embedding, limit)
            
            cur.execute(query, params)
            results = cur.fetchall()
            cur.close()
            conn.close()
    
            return results

        except Exception as e:
            print("Error querying the database:", e)
    
    # Hauptverarbeitungsschritt für PDF-Ordner
    def process_pdfs(self, pdf_folder):
        self.set_pdf_files(pdf_folder)
        filenames_array = []
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            # SQL-Abfrage: Eindeutige Filenames als Array abfragen
            cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding;")
            print("Index dropped for performance increase.")
            cur.execute("SELECT array_agg(DISTINCT filename) FROM chunks;")
            filenames_array = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print("Error extracting filenames from database:", e)

        for pdf_path in self.pdf_files:

            filename=os.path.basename(pdf_path)
            if filename in filenames_array:
                print(f"File {filename} already processed, skipping...")
                continue

            print(f"Processing file: {pdf_path}")
            text = self.extract_text(pdf_path)
            chunks = self.chunk_text(text, self.max_tokens)

            for i, chunk in enumerate(chunks):
                keywords = self.get_keywords_with_gpt(chunk)

                # **Separates Embedding für Chunk-Text und Keywords**
                chunk_embedding = self.embedding_client.embed_query(chunk)

                # **Datenbank speichern**
                self.insert_into_db(
                    filename=filename,
                    chunk_number=i + 1,
                    chunk_text=chunk, 
                    keywords=keywords,
                    embedding=chunk_embedding
                )

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                        ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                        """)
            conn.commit()
            cur.close()
            conn.close()
            print("Index created succesfully.")
        except Exception as e:
            print("Error creating index:", e)
                


        print("All PDFs have been processed and stored in the database.")


if __name__ == "__main__":
 
    pdf_folder = "test_pdfs"
    processor = PDFProcessor()
    processor.process_pdfs(pdf_folder)
    
    """
    processor = PDFProcessor()
    results = processor.search_chunks("Wie Verbinde ich Acoustimass 15 mit meinem Fernseher?", keywords_filter=["Bose", "Acoustimass"])
    for result in results:
        print(result)
    """