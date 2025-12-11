# Readability Influences Retrieval

### Website Link: https://ronak-wani.github.io/Readability-Influences-Retrieval/

## Getting started

Install Ollama based on your operating system (https://ollama.com/download):

Linux Command: 
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installing ollama, pull the following models:

- snowflake-artic-embed-text
- nomic-embed-text
- granite-embedding
- embeddinggemma
- qwen3-embedding
- snowflake-artic-embed-text2

Command: 
```bash
ollama pull <model-name>
```

Install Python 3.12 (https://www.python.org/downloads/release/python-31211/)

Then create a virtual environment:

```bash
python -m venv venv
```

MacOS/Linux: 

```bash
source venv/bin/activate
```
Windows:

```bash
 .\venv\Scripts\activate
```
The clone this repository:

```bash
git clone https://github.com/username/repository.git
```

Install the dependencies from requirements.txt using the following commands:
```bash
 pip install -r requirements.txt
```
Run the program from the main.py

```bash
 python main.py
```
It will create a ChromaDB folder, which is a persistent vector database instance containing the vector embeddings. 
It will also create a folder for each model, with each folder containing the results for BM25, TF-IDF, Cosine, Dot Product, and Euclidean Distance.




For the notebook files generated_readability and generate answers. These can be added to a Python notebook, or Colab can be used, which is what we used. Make sure to pip install the imports. For the generated answers, depending on the computer power available, this can take a long time as we used an LLM with 27b parameters. A smaller ollam model can be used, or we recommend using our generated answers file. Using this file you can test out the generated readability.
