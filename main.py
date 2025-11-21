from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class RAG:
    def __init__(self, type):
        type = self.type

    def data_loader(self):

        pass

    def vectordb(self):
        vectorstore = Chroma(
            # documents=documents,
            collection_name="readability-rag",
            embedding=OllamaEmbeddings(model='nomic-embed-text', show_progress=True),
            persist_directory="./chroma_db",
        )
        vectorstore.persist()

    def semantic_search(self):
        match type:
            case "cosine":
                pass
            case "euclidean":
                pass
            case "dot_product":
                pass
            case _:
                print("Type of embedding similarity needs to be specified")

        pass

    def evaluation(self):
        pass

if __name__=="__main__":
    RAG rag_cosine = RAG("cosine")
    RAG rag_euclidean = RAG("euclidean")
    RAG rag_dot_product = RAG("dot_product")