# -*- coding: utf-8 -*-
"""generate_answers.ipynb
"""

!curl -s https://ollama.ai/install.sh | sh
!ollama serve &>/dev/null &
!ollama pull gemma3:27b


import json
import requests
from tqdm.auto import tqdm



with open("/content/combined_passages.json", "r", encoding="utf-8") as f:
    combined_passages = json.load(f)

with open("/content/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

answers = {}

for title, text_levels in tqdm(combined_passages.items(), desc="Articles"):
    questions_for_titles = questions.get(title)

    answers[title] = {}

    for text_level, text in text_levels.items():
        text_answers = {}

        for question_level, question in questions_for_titles.items():
            prompt = f"""Article:\n{text}\n
            nQuestion:\n{question}
            \n\nAnswer briefly."""

            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "gemma3:27b", "prompt": prompt, "stream": False},
                timeout=120,
            ).json()

            answer_level = question_level.replace("-Q", "-A")
            text_answers[answer_level] = resp.get("response", "").strip()

        answers[title][text_level] = text_answers


with open("/content/answers.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, ensure_ascii=False, indent=2)

