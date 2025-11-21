import json
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import OllamaEmbeddings

class RAG:
    type = None
    def __init__(self, type):
        type = self.type

    def data_loader(self):
        ids=[]
        loader = JSONLoader(
            file_path="./combined_passages.json",
            jq_schema='to_entries[] | {title: .key, level: "Adv", text: .value["Adv-Txt"]} , '
                      '{title: .key, level: "Int", text: .value["Int-Txt"]} , '
                      '{title: .key, level: "Ele", text: .value["Ele-Txt"]}',
            text_content=False,
        )
        docs = loader.load()
        # print(docs[0])
        for i in range(len(docs)):
            data = json.loads(docs[i].page_content)
            title = data.get("title")
            level = data.get("level")
            # print("Title:", title)
            # print("Level:", level)
            id = title + "-" + level
            # print(id)
            ids.append(id)
        return docs, ids

    def vectordb(self, docs, ids):
        vectorstore = Chroma(
            collection_name="readability-rag",
            embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
            persist_directory="./chroma_db",
        )
        vectorstore.add_documents(documents=docs, ids=ids)


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
    docs, ids = rag_cosine.data_loader()
    rag_cosine.vectordb(docs, ids)
    rag_cosine.semantic_search()
    rag_euclidean = RAG("euclidean")
    rag_dot_product = RAG("dot_product")