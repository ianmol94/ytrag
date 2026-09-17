# YouTube RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about the content of a YouTube video.

## Features

* Fetches YouTube video transcripts
* Splits transcripts into smaller chunks
* Generates embeddings using Hugging Face
* Stores and retrieves relevant chunks using FAISS
* Uses Groq LLM for answer generation
* Answers questions based on the retrieved transcript context

## Tech Stack

* Python
* LangChain
* Groq
* Hugging Face Sentence Transformers
* FAISS
* YouTube Transcript API
* uv

## RAG Pipeline

```text
YouTube Video
      ↓
Transcript
      ↓
Text Splitting
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Retriever
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Answer
```

## Setup

Clone the repository:

```bash
git clone https://github.com/ianmol94/ytrag.git
cd ytrag
```

Install dependencies using `uv`:

```bash
uv sync
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Run the application:

```bash
uv run python main.py
```

## Example

Provide a YouTube video ID and ask questions about its transcript.

Example:

```text
Question: Can you summarize the video?

Answer: ...
```

## Project Structure

```text
ytrag/
├── main.py
├── ytragchat.py
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

## Future Improvements

* Add a web-based chat interface
* Support arbitrary YouTube URLs
* Add conversation history
* Replace FAISS with a hosted vector database
* Add streaming responses
* Deploy the application

## Author

**Anmol Singh**

GitHub: https://github.com/ianmol94
