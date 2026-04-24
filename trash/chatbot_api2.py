# importing libraries
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA          
from langchain_core.prompts import PromptTemplate
from warnings import filterwarnings
filterwarnings("ignore")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()


# request / response models 
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str


# CORE API LOGIC: load resources at startup, handle queries, and clean up on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup and clean up on shutdown."""
    print("Starting up the travel chatbot API...")

    # Pinecone setup
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in environment variables.")
    Pinecone(api_key=pinecone_api_key)

    # Embeddings — same model used in pipeline.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Connect to the existing Pinecone index (data already uploaded by pipeline.py)
    index_name = "chatbot"
    vecstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=gemini_api_key,
    )

    # Retriever & prompt
    retriever = vecstore.as_retriever(search_kwargs={"k": 3})
    prompt_template = '''أنت رفيق المساعد الذكى فى السفر. استخدم المعلومات التالية للإجابة على سؤال المستخدم باللغة العربية.
إذا كانت المعلومات المقدمة لا تحتوي على إجابة مباشرة، فحاول أن تستنتج أقرب معلومة أو قل أن هذه المعلومة غير متوفرة في قاعدة بياناتنا، مع اقتراح ما هو متاح 
المعلومات المتاحة:
{context}

السؤال: {question}
الرد:'''
    PROMPT = PromptTemplate.from_template(prompt_template)

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
    )

    app.state.qa_chain = qa_chain
    print("Travel chatbot API is ready to receive queries.")

    yield  # app runs here

    # shutdown logic (if needed) goes here
    print("Shutting down the travel chatbot API.")


# app 
app = FastAPI(
    title="Travel Chatbot API",
    description="API for querying the travel chatbot",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def ask(request: QueryRequest):
    qa_chain = getattr(app.state, "qa_chain", None)
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="النظام لم يكتمل تحميله بعد، حاول مرة أخرى.",
        )
    response = qa_chain.invoke({"query": request.query})
    answer = response["result"].strip()
    return QueryResponse(response=answer)
