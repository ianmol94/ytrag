import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


## Step 1a : Indexing (document ingestion)

video_id = "Gfr50f6ZBvo"

try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    # Convert transcript snippets into plain text
    transcript = " ".join(
        snippet.text for snippet in transcript_list
    )

    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
    
transcript_list

## Step 1b : Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

len(chunks)

## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

## Step 2 : Retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever.invoke('What is deepmind?')

## Step 3 : Augumentation

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

question = "is the topic of philosophy discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)


# joining all page content into a single string 
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

## Step 4 : Generation 

answer = llm.invoke(final_prompt)
print("")
print("======LLM response=====")
print("")
print(answer.content)

## Building a Chain 

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


def format_docs(retrieved_docs):
    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )
    return context_text


parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser 
print("")
print("====summarizing the video====")
print("")
response = main_chain.invoke('can you summarize the video ?')
print(response)
