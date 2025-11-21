import os, json

base = "/Users/matthewkiszla/Downloads/Readability-Influences-Retrieval-main/Texts-SeparatedByReadingLevel"
levels = ["Adv-Txt", "Int-Txt", "Ele-Txt"]

data = {}

def json_create():
    for level in levels:
        path = os.path.join(base, level)
        for fname in os.listdir(path):
            if fname.endswith(".txt"):
                name = os.path.splitext(fname)[0]
                article = name.rsplit("-", 1)[0]
                if article not in data:
                    data[article] = {}
                with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                    text = f.read().lstrip("\ufeff").strip()
                    data[article][level] = text

    with open("combined_passages.json", "w", encoding="utf-8") as out:
        json.dump(data, out, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    json_create()
