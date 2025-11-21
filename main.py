from unittest import case

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import OllamaEmbeddings

class RAG:
    type = None
    def __init__(self, type):
        type = self.type

    def data_loader(self):
        loader = JSONLoader(
            file_path="./combined_passages.json",
            jq_schema='to_entries[] | {title: .key, level: "Adv", text: .value["Adv-Txt"]} , '
                      '{title: .key, level: "Int", text: .value["Int-Txt"]} , '
                      '{title: .key, level: "Ele", text: .value["Ele-Txt"]}',
            text_content=False,
        )
        docs = loader.load()
        # print(docs[0])
        return docs

    def vectordb(self, docs):
        vectorstore = Chroma(
            # documents=docs,
            collection_name="readability-rag",
            # embedding=OllamaEmbeddings(model='nomic-embed-text'),
            persist_directory="./chroma_db",
        )
        # vectorstore.persist()

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
    rag_cosine = RAG("cosine")
    docs = rag_cosine.data_loader()
    rag_cosine.vectordb(docs)
    rag_euclidean = RAG("euclidean")
    rag_dot_product = RAG("dot_product")