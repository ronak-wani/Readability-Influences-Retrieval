import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import OllamaEmbeddings

class RAG:
    type = None
    llm = None
    def __init__(self, type, llm):
        self.type = type
        self.llm = llm

    def data_loader(self):
        ids=[]
        loader = JSONLoader(
            file_path="./combined_passages.json",
            jq_schema='to_entries[] | {title: .key, level: "Adv", text: .value["Adv-Txt"]} , '
                      '{title: .key, level: "Int", text: .value["Int-Txt"]} , '
                      '{title: .key, level: "Ele", text: .value["Ele-Txt"]}',
            content_key="text",
            metadata_func=lambda record, metadata: {
                "title": record["title"],
                "level": record["level"]
            },
            text_content=True,
        )
        docs = loader.load()
        print(docs[0])
        # print(docs[0].page_content)
        for i in range(len(docs)):
            # data = json.loads(docs[i].metadata)
            title = docs[i].metadata.get("title")
            level = docs[i].metadata.get("level")
            # print("Title:", title)
            # print("Level:", level)
            id = title + "-" + level
            # print(id)
            ids.append(id)
        return docs, ids

    def vectordb(self, docs, ids):
        vectorstore = Chroma(
            collection_name="readability-rag",
            embedding_function=OllamaEmbeddings(model='snowflake-arctic-embed', base_url="http://localhost:11434"),
            persist_directory="./chroma_db",
        )
        print(f"Adding {len(docs)} documents to vector store...")

        batch_size = 2
        for i in tqdm(range(0, len(docs), batch_size), desc="Adding documents"):
            batch_docs = docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            vectorstore.add_documents(documents=batch_docs, ids=batch_ids)

        retriever = vectorstore.as_retriever()
        return retriever

    def semantic_search(self):
        docs, ids = rag_cosine.data_loader()
        retriever = rag_cosine.vectordb(docs, ids)
        rag_cosine.rag_chain(question, retriever)
        rag_cosine.evaluation()
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

    def rag_chain(self,question, retriever):
        print("\n########\nAfter RAG\n")
        prompt = """Answer the question based only on the provided context:
        {context}
        Question: {question}
        """
        after_rag_prompt = ChatPromptTemplate.from_template(prompt)
        after_rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | after_rag_prompt
                | self.llm
                | StrOutputParser()
        )
        response = after_rag_chain.invoke({"question": question})
        return {"response": response}

    def evaluation(self):
        pass

if __name__=="__main__":
    rag_cosine = RAG("cosine", "phi4")
    rag_cosine.semantic_search()
    rag_euclidean = RAG("euclidean", "phi4")
    rag_dot_product = RAG("dot_product", "phi4")