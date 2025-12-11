import json
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
import os

TOP_K = 3

class RAG:
    type = None
    llm = None
    docs = []
    ids = []
    vectorstore = None
    bm25 = None
    tfidf = None
    search = None
    results = None

    def __init__(self, type, llm):
        self.type = type
        self.llm = llm
        self.precision_list = [0] * 3
        self.recall_list = [0] * 3
        self.Ele_Q_list = [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]  # ["Ele_Doc_Rank", "Int_Doc_Rank", "Adv_Doc_Rank"]

        self.Int_Q_list = [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]] # ["Ele_Doc_Rank", "Int_Doc_Rank", "Adv_Doc_Rank"]

        self.Adv_Q_list = [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]  # ["Ele_Doc_Rank", "Int_Doc_Rank", "Adv_Doc_Rank"]
        self.docs, self.ids = self.data_loader()
        self.retriever = self.vectordb()
        self.bm25 = BM25Retriever.from_documents(self.docs)
        self.bm25.k = TOP_K
        self.tfidf = TFIDFRetriever.from_documents(self.docs)
        self.tfidf.k = TOP_K
        self.folder_name = self.create_output_folder()
        self.semantic_search()
        self.results = self.evaluation()

    def create_output_folder(self):
        folder_name = f"{self.llm}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created output folder: {folder_name}")
        else:
            print(f"Output folder already exists: {folder_name}")

        return folder_name

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
        # print(docs[0])
        # print(docs[0].page_content)
        for i in range(len(docs)):
            title = docs[i].metadata.get("title")
            level = docs[i].metadata.get("level")
            # print("Title:", title)
            # print("Level:", level)
            id = title + "-" + level
            # print(id)
            ids.append(id)
        return docs, ids

    def vectordb(self):
        vectorstore = Chroma(
            collection_name="readability-rag-" + self.llm,
            embedding_function = OllamaEmbeddings(
                model=self.llm,
                base_url="http://localhost:11434",
                num_ctx=512,
            ),
            persist_directory="./chroma_db_" + self.llm.replace("-", "_"),
        )
        self.vectorstore = vectorstore
        existing_doc_count = vectorstore._collection.count()

        if existing_doc_count > 0:
            print(f"ChromaDB already has {existing_doc_count} documents.")
            print("Skipping embedding step and using existing vectorDB.")

        else:
            print(f"Adding {len(self.docs)} documents to vector store...")

            batch_size = 2
            for i in tqdm(range(0, len(self.docs), batch_size), desc="Adding documents"):
                batch_docs = self.docs[i:i + batch_size]
                batch_ids = self.ids[i:i + batch_size]
                vectorstore.add_documents(documents=batch_docs, ids=batch_ids)

        retriever = vectorstore.as_retriever()
        return retriever

    def semantic_search(self):
        match self.type:
            case "cosine":
                self.search = self.retriever
            case "euclidean":
                self.search = self.retriever
            case "dot_product":
                self.search = self.retriever
            case "bm25":
                self.search = self.bm25
            case "tfidf":
                self.search = self.tfidf
            case _:
                print("Type of embedding similarity needs to be specified")

        pass

    def rag_chain(self, questions_file_path):
        responses = []
        prompt = """Answer the question based only on the provided context:
        {context}
        Question: {question}
        """

        rag_prompt = ChatPromptTemplate.from_template(prompt)

        with open(questions_file_path, "r") as file:
            data = json.load(file)

        for topic, questions in data.items():
            for q_type, question in questions.items():
                print(f"Question: {question}")
                response = self.search.invoke(f"{question}")
                print(f"Response: {response}")

            print()

            return responses

    
    def evaluation(self):
            with open("./questions.json", "r") as f:
                questions = json.load(f)

            embeddings = None
            if self.type in ["cosine", "euclidean", "dot_product"]:
                collection = self.vectorstore._collection
                all_ids = collection.get()["ids"]
                data = collection.get(include=["embeddings"], ids=all_ids)
                embeddings = np.array(data["embeddings"])

            results = {}

            for title, questions in tqdm(questions.items(), desc=f"Testing {self.type}"):
                
                title_result = {}
                retrieved_docs=[]

                for query_level, question in questions.items():
                    if self.type == "bm25":
                        docs = self.bm25.invoke(question)[:3]
                        retrieved_docs = [(d, 0.0) for d in docs]

                    elif self.type == "tfidf":
                        docs = self.tfidf.invoke(question)[:3]
                        retrieved_docs = [(d, 0.0) for d in docs]

                    else:
                        query_vector = np.array(self.vectorstore._embedding_function.embed_query(question))
                        if self.type == "cosine":
                            a = query_vector
                            b = embeddings
                            a_norm = a / (np.linalg.norm(a))
                            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True))
                            similarities = np.dot(b_norm, a_norm)
                            order = np.argsort(-similarities)[:3]
                            retrieved_docs = [(self.docs[int(i)], float(similarities[i])) for i in order]
                        elif self.type == "dot_product":
                            similarities = embeddings @ query_vector
                            order = np.argsort(-similarities)[:3]
                            retrieved_docs = [(self.docs[int(i)], float(similarities[i])) for i in order]
                        elif self.type == "euclidean":
                            distances = np.linalg.norm(embeddings - query_vector, axis=1)
                            order = np.argsort(distances)[:3]
                            retrieved_docs = [(self.docs[int(i)], float(distances[i])) for i in order]

                    top_k_docs = []
                    for rank, (document, score) in enumerate(retrieved_docs):
                        top_k_docs.append({
                                "rank": rank+1,
                                "title": document.metadata.get("title"),
                                "level": document.metadata.get("level"),
                                "score": score,
                            })
                        if query_level == "Ele-Q":
                            if document.metadata.get("level") == "Ele" and document.metadata.get("title") == title:
                                self.Ele_Q_list[0][rank] += 1
                            elif document.metadata.get("level") == "Int" and document.metadata.get("title") == title:
                                self.Ele_Q_list[1][rank] += 1
                            elif document.metadata.get("level") == "Adv" and document.metadata.get("title") == title:
                                self.Ele_Q_list[2][rank] += 1
                        elif query_level == "Int-Q":
                            if document.metadata.get("level") == "Ele" and document.metadata.get("title") == title:
                                self.Int_Q_list[0][rank] += 1
                            elif document.metadata.get("level") == "Int" and document.metadata.get("title") == title:
                                self.Int_Q_list[1][rank] += 1
                            elif document.metadata.get("level") == "Adv" and document.metadata.get("title") == title:
                                self.Int_Q_list[2][rank] += 1
                        elif query_level == "Adv-Q":
                            if document.metadata.get("level") == "Ele" and document.metadata.get("title") == title:
                                self.Adv_Q_list[0][rank] += 1
                            elif document.metadata.get("level") == "Int" and document.metadata.get("title") == title:
                                self.Adv_Q_list[1][rank] += 1
                            elif document.metadata.get("level") == "Adv" and document.metadata.get("title") == title:
                                self.Adv_Q_list[2][rank] += 1

                    num_relevant_retrieved = 0
                        
                    for document in top_k_docs:
                        if document.get("title") == title:
                            num_relevant_retrieved += 1

                    precision = num_relevant_retrieved / 3

                    if query_level == "Ele-Q":
                        self.precision_list[0] += precision
                    if query_level == "Int-Q":
                        self.precision_list[1] += precision
                    if query_level == "Adv-Q":
                        self.precision_list[2] += precision

                    recall = num_relevant_retrieved / 3
                    

                    level_ranks = {"Adv": None, "Int": None, "Ele": None}
                    for rank, document in enumerate(top_k_docs):
                        if document.get("title") == title:
                            lvl = document.get("level")
                            level_ranks[lvl] = rank + 1

                    title_result[query_level] = {
                        "question": question,
                        "topic": title,
                        "query_level": query_level,
                        "top_k_docs": top_k_docs,
                        "metrics": {
                            "precision_at_k=3": precision,
                            "recall_at_k=3": recall,
                            "num_relevant": 3,
                            "level_ranks": level_ranks,
                        },

                    }

                results[title] = title_result

            output = {
                "config": {
                    "model": self.llm,
                    "metric": self.type,
                    "k": 3,
                    "Ele-Q" : {
                        "Count of relevant Ele doc at rank 1 ": self.Ele_Q_list[0][0],
                        "Count of relevant Ele doc at rank 2 ": self.Ele_Q_list[0][1],
                        "Count of relevant Ele doc at rank 3 ": self.Ele_Q_list[0][2],
                        "Weighted Rank Score For Ele Doc": ((self.Ele_Q_list[0][0]*3) + (self.Ele_Q_list[0][1]*2) + (self.Ele_Q_list[0][2]*1))/(189 * 3),

                        "Count of relevant Int doc at rank 1 ": self.Int_Q_list[0][0],
                        "Count of relevant Int doc at rank 2 ": self.Int_Q_list[0][1],
                        "Count of relevant Int doc at rank 3 ": self.Int_Q_list[0][2],
                        "Weighted Rank Score For Int Doc": ((self.Int_Q_list[0][0] * 3) + (self.Int_Q_list[0][1] * 2) + (self.Int_Q_list[0][2] * 1)) / (189 * 3),

                        "Count of relevant Adv doc at rank 1 ": self.Adv_Q_list[0][0],
                        "Count of relevant Adv doc at rank 2 ": self.Adv_Q_list[0][1],
                        "Count of relevant Adv doc at rank 3 ": self.Adv_Q_list[0][2],
                        "Weighted Rank Score For Adv Doc": ((self.Adv_Q_list[0][0] * 3) + (self.Adv_Q_list[0][1] * 2) + (self.Adv_Q_list[0][2] * 1)) / (189 * 3),
                    },
                    "Int-Q": {
                        "Count of relevant Ele doc at rank 1 ": self.Ele_Q_list[1][0],
                        "Count of relevant Ele doc at rank 2 ": self.Ele_Q_list[1][1],
                        "Count of relevant Ele doc at rank 3 ": self.Ele_Q_list[1][2],
                        "Weighted Rank Score For Ele Doc": ((self.Ele_Q_list[1][0]*3) + (self.Ele_Q_list[1][1]*2) + (self.Ele_Q_list[1][2]*1))/(189 * 3),

                        "Count of relevant Int doc at rank 1 ": self.Int_Q_list[1][0],
                        "Count of relevant Int doc at rank 2 ": self.Int_Q_list[1][1],
                        "Count of relevant Int doc at rank 3 ": self.Int_Q_list[1][2],
                        "Weighted Rank Score For Int Doc": ((self.Int_Q_list[1][0] * 3) + (self.Int_Q_list[1][1] * 2) + (self.Int_Q_list[1][2] * 1)) / (189 * 3),

                        "Count of relevant Adv doc at rank 1 ": self.Adv_Q_list[1][0],
                        "Count of relevant Adv doc at rank 2 ": self.Adv_Q_list[1][1],
                        "Count of relevant Adv doc at rank 3 ": self.Adv_Q_list[1][2],
                        "Weighted Rank Score For Adv Doc": ((self.Adv_Q_list[1][0] * 3) + (self.Adv_Q_list[1][1] * 2) + (self.Adv_Q_list[2][2] * 1)) / (189 * 3),
                    },
                    "Adv-Q": {
                        "Count of relevant Ele doc at rank 1 ": self.Ele_Q_list[2][0],
                        "Count of relevant Ele doc at rank 2 ": self.Ele_Q_list[2][1],
                        "Count of relevant Ele doc at rank 3 ": self.Ele_Q_list[2][2],
                        "Weighted Rank Score For Ele Doc": ((self.Ele_Q_list[2][0]*3) + (self.Ele_Q_list[2][1]*2) + (self.Ele_Q_list[2][2]*1))/(189*3),

                        "Count of relevant Int doc at rank 1 ": self.Int_Q_list[2][0],
                        "Count of relevant Int doc at rank 2 ": self.Int_Q_list[2][1],
                        "Count of relevant Int doc at rank 3 ": self.Int_Q_list[2][2],
                        "Weighted Rank Score For Int Doc": ((self.Int_Q_list[2][0] * 3) + (self.Int_Q_list[2][1] * 2) + (self.Int_Q_list[2][2] * 1)) / (189 * 3),

                        "Count of relevant Adv doc at rank 1 ": self.Adv_Q_list[2][0],
                        "Count of relevant Adv doc at rank 2 ": self.Adv_Q_list[2][1],
                        "Count of relevant Adv doc at rank 3 ": self.Adv_Q_list[2][2],
                        "Weighted Rank Score For Adv Doc": ((self.Adv_Q_list[2][0] * 3) + (self.Adv_Q_list[2][1] * 2) + (self.Adv_Q_list[2][2] * 1)) / (189 * 3),
                    },

                    "Precision @ Ele": f"{(self.precision_list[0]/189):.2f}",
                    "Precision @ Int": f"{(self.precision_list[1]/189):.2f}",
                    "Precision @ Adv": f"{(self.precision_list[2]/189):.2f}",
                    "Recall @ Ele": f"{(self.precision_list[0]/189):.2f}",
                    "Recall @ Int": f"{(self.precision_list[1]/189):.2f}",
                    "Recall @ Adv": f"{(self.precision_list[2]/189):.2f}",
                },
                "results": results,
            }

            filename = os.path.join(self.folder_name, f"results_{self.llm}_{self.type}.json")

            with open(filename, "w") as f:
                json.dump(output, f, indent=4)

            return output

if __name__=="__main__":
    models = ["snowflake-arctic-embed", "snowflake-arctic-embed2", "nomic-embed-text", "granite-embedding",
              "embeddinggemma", "qwen3-embedding"]
    types = ["cosine", "euclidean", "dot_product", "bm25", "tfidf"]
    for m in models:
        for t in types:
            RAG(t, m)
