# Mini-RAG Telegram Bot with Vision

A Telegram bot that answers questions from your own documents (RAG) and can also describe uploaded images.
Runs fully locally using Ollama, sentence-transformers, and CLIP.

---

## What this bot does

- 📚 **RAG (Retrieval-Augmented Generation)**
  Answers questions strictly from your local text/markdown files.

- 🧠 **Local LLM (via Ollama)**
  No OpenAI or cloud APIs required.

- 📸 **Image understanding**
  Describes uploaded images using a CLIP vision model.

- 💬 **Telegram interface**
  Simple commands, fast responses, memory caching.

---

## Project structure

```
.
├── app.py
├── rag.py
├── prompts.py
├── vision.py
├── config.py
├── data/
├── rag.db
└── requirements.txt
```

---

## How to run locally (Python)

### Prerequisites
- Python 3.9+
- Telegram bot token
- Ollama installed and running

Pull model:
```
ollama pull llama3
```

### Install dependencies
```
pip install -r requirements.txt
```

### Configure
Edit `config.py` and add your bot token.

### Run
```
python app.py
```

---

## Telegram commands

- `/start`
- `/ask <question>`
- `/image`
- Send a photo

---

## Example Telegram interactions

**/ask**
```
User: /ask What is this project?
Bot: Answers based on indexed documents
```

**Image upload**
```
User uploads image
Bot: A wide photo showing a street
Tags: Street, Building, Outdoor
```

---

## Models and APIs used

- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- LLM: llama3 via Ollama
- Vision: openai/clip-vit-base-patch32

---

## System design

```
Telegram User
     |
     v
Python Bot
  |      |
 RAG   Vision
  |      |
 Ollama CLIP
```

---

## License
MIT
