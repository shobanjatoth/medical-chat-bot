# import os

# from dotenv import load_dotenv

# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse, PlainTextResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles

# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_core.prompts import ChatPromptTemplate

# from src.helper import download_hugging_face_embeddings
# from src.prompt import system_prompt


# # Load environment variables
# load_dotenv()

# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# # FastAPI app
# app = FastAPI()


# # Static files
# app.mount(
#     "/static",
#     StaticFiles(directory="static"),
#     name="static"
# )


# # Templates
# templates = Jinja2Templates(directory="templates")


# # Embeddings
# embeddings = download_hugging_face_embeddings()


# # Pinecone index
# index_name = "medical-chat-bot"


# # Vector store
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )


# # Retriever
# retriever = docsearch.as_retriever(
#     search_type="similarity",
#     search_kwargs={"k": 3}
# )


# # Gemini model
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.3,
#     google_api_key=GOOGLE_API_KEY
# )


# # Prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}")
#     ]
# )


# # QA chain
# question_answer_chain = create_stuff_documents_chain(
#     llm,
#     prompt
# )


# # RAG chain
# rag_chain = create_retrieval_chain(
#     retriever,
#     question_answer_chain
# )


# # Home route
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):

#     return templates.TemplateResponse(
#         "chat.html",
#         {"request": request}
#     )


# # Chat route
# @app.post("/get", response_class=PlainTextResponse)
# async def chat(msg: str = Form(...)):

#     try:

#         response = rag_chain.invoke(
#             {"input": msg}
#         )

#         answer = response["answer"]

#         return answer

#     except Exception as e:

#         print("Error:", str(e))

#         return f"Error: {str(e)}"


import os

from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(directory="templates")


# =========================================================
# EMBEDDINGS
# =========================================================

embeddings = download_hugging_face_embeddings()


# =========================================================
# VECTOR STORE
# =========================================================

index_name = "medical-chat-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# =========================================================
# RETRIEVER
# =========================================================

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# =========================================================
# QA CHAIN
# =========================================================

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)


# =========================================================
# RAG CHAIN
# =========================================================

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# =========================================================
# ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "chat.html",
        {"request": request}
    )


@app.post("/get", response_class=PlainTextResponse)
async def chat(msg: str = Form(...)):

    try:

        response = rag_chain.invoke(
            {"input": msg}
        )

        answer = response["answer"]

        return answer

    except Exception as e:

        print("ERROR:", str(e))

        return f"Error: {str(e)}"


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {"status": "ok"}