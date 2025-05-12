
import os
import sys
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from langchain_core.documents import Document
import base64
import yaml
import faiss
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.base import Embeddings
import nltk
import openai  # Für GPT-API
from openai import AzureOpenAI
nltk.download('punkt_tab')
os.chdir(sys.path[0])
import yake

class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = AzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_base, api_version="2024-08-01-preview")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

class PDFTextExtractor:
    def __init__(self, config_path):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        self.gptTurboEndpoint = config["gptTurboEndpoint"]
        self.gptTurboKey = config["gptTurboKey"]
        self.embedding_model = "text-embedding-ada-002"
        self.embeddingEndpoint = config["embeddingEndpoint"]
        self.embeddingKey = config["embeddingKey"]
        self.endpoint = config["endPoint"]
        self.key = config["apiKey"]
        self.client = DocumentIntelligenceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))
        self.embedding_client = AzureOpenAIEmbeddings(
            api_key=self.embeddingKey,
            api_base=self.embeddingEndpoint,
            model=self.embedding_model
        )
        self.max_tokens = config["maxTokens"]
        self.pdfArray = []
        self.DocList = []

    def extract_text(self, pdf_path):
        with open(pdf_path, "rb") as f:
            base64_encoded_pdf = base64.b64encode(f.read()).decode("utf-8")

            poller = self.client.begin_analyze_document("prebuilt-layout", AnalyzeDocumentRequest(bytes_source=base64_encoded_pdf))
        
        result = poller.result()
        text = ""
        for page in result.pages:
            for line in page.lines:
                text += line.content + " "
        return text

    def extract_text_from_pdfs(self, pdf_paths):
        texts = []
        for pdf_path in pdf_paths:
            texts.append(self.extract_text(pdf_path))   
        return texts
    
    # def get_keywords_with_gpt(self, chunk):
    #     prompt = (
    #         f"Create a list of 5 keywords that summarize the following text:\n\n"
    #         f"Text: {chunk}\n\n"
    #         f"Please return the keywords as a comma-separated list."
    #     )

    #     client = AzureOpenAI(
    #         api_key=self.gptTurboKey,
    #         azure_endpoint=self.gptTurboEndpoint,
    #         api_version="2024-08-01-preview"
    #     )

    #     chat_completion = client.chat.completions.create(
    #         model="gpt-35-turbo", 
    #         messages=[{"role": "user", "content": prompt}],
    #         max_tokens=50,
    #         temperature=0.5
    #     )

    #     if chat_completion and chat_completion.choices:
    #         message = chat_completion.choices[0].message
    #         if message and message.content:
    #             return message.content.strip()
    #     return "No keywords generated"

    def get_keywords_with_yake(self, chunk, num_keywords=5):
        extractor = yake.KeywordExtractor(
            lan="en",              
            n=3,                   
            dedupLim=0.9,          
            windowsSize=1,         
            top=num_keywords      
        )
    
        # Keywords extrahieren
        keywords = extractor.extract_keywords(chunk)
    
        # Extrahierte Keywords zurückgeben (nur die Keywords, nicht die Scores)
        extracted_keywords = [kw for kw, score in keywords]
        return ", ".join(extracted_keywords)

    def save_text_to_file(self, text, output_path):
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

    def chunk_text(self, text, max_tokens, overlap = 5):
        sentences = nltk.sent_tokenize(text)
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:]  # Overlap
                current_tokens = sum(len(word.split()) for word in current_chunk)
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def set_pdfs(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    self.pdfArray.append(os.path.join(root, file))
                    print(self.pdfArray)

    def process_pdfs_txt(self, config_path, pdf_folder, output_folder):
        extractor = PDFTextExtractor(config_path)
        extractor.set_pdfs(pdf_folder)
        os.makedirs(output_folder, exist_ok=True)

        for pdf_path in extractor.pdfArray:
            print(f"Verarbeite Datei: {pdf_path}")
            
            text = extractor.extract_text(pdf_path)
            
            chunks = extractor.chunk_text(text, extractor.max_tokens)    
            
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]  
            output_path = os.path.join(output_folder, f"{base_name}_chunks.txt")
            
            with open(output_path, "w", encoding="utf-8") as output_file:
                for i, chunk in enumerate(chunks):
    
                    keywords = self.get_keywords_with_yake(chunk)   
                    metadata = {
                        "Source": pdf_path,
                        "Chunk Number": i + 1,
                        "Keywords": ", ".join([kw for kw, _ in keywords])  
                    }
                    
                    output_file.write(f"--- Chunk {i+1} ---\n")
                    for key, value in metadata.items():
                        output_file.write(f"{key}: {value}\n")
                    output_file.write("\nText:\n")
                    output_file.write(chunk + "\n\n")
            
            print(f"Alle Chunks der PDF wurden in '{output_path}' gespeichert.")
    
    def process_pdfs(self, pdf_folder):
        self.set_pdfs(pdf_folder)

        for pdf_path in self.pdfArray:
            print(f"Verarbeite Datei: {pdf_path}")
            
            text = self.extract_text(pdf_path)
            
            chunks = self.chunk_text(text, self.max_tokens)        

            for i, chunk in enumerate(chunks):
                keywords_string = self.get_keywords_with_yake(chunk)
                keywords_list = keywords_string.split(", ") 
                metadata = {
                    "Source": pdf_path,
                    "Chunk Number": i + 1,
                    "Keywords": ", ".join(keywords_list)
                }
                
                self.DocList.append(Document(page_content=chunk,metadata=metadata))
                print(f"Chunk {i+1} verarbeitet.")
        print(f"Alle PDFs verarbeitet.")

    def setup_faiss(self, fileName = "VectorStore"):
        uuids = [str(uuid4()) for _ in range(len(self.DocList))]
        index = faiss.IndexFlatL2(1536)  # 1536 dimensions for text-embedding-ada-002
        vector_store = FAISS(
            embedding_function=self.embedding_client,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        vector_store.add_documents(documents=self.DocList, ids=uuids)
        print("Vektor-Store wurde erstellt.")
        vector_store.save_local(fileName)
        print("Vektor-Store wurde gesichert.")
        return vector_store

def test_vector_store_persistence():
    print(os.getcwd())    # need to "import os" first
    config_path = "config.yaml"
    pdf_folder = "pdf-manuals"
    output_folder = "pdf_chunked"
    extractor = PDFTextExtractor(config_path)
    
    # PDFs verarbeiten und Chunks erstellen
    extractor.process_pdfs(pdf_folder)
    vector_store = extractor.setup_faiss(fileName="completeStorage")
    #vector_store.save_local("test.faiss")
    print("Vektor-Store wurde erfolgreich erstellt und gespeichert.")
    
    # Vektor-Store laden und abfragen
    vector_store_loaded = FAISS.load_local(
        "completeStorage", 
        embeddings=extractor.embedding_client,
        allow_dangerous_deserialization=True
    )
    query = "What to do if no noise reduction"
    results = vector_store_loaded.similarity_search(query, k=4)
    print(f"Ergebnisse für die Abfrage: {query}")
    for res in results:
        print(f"* {res.page_content}")

if __name__ == "__main__":
    test_vector_store_persistence()

