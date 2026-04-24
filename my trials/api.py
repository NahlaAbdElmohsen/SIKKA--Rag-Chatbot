# importing libraries
import os
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
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


# entity extraction helper 
def extract_route_entities(query: str, llm: ChatGoogleGenerativeAI) -> dict:
    """
    Ask the LLM to extract origin and destination from the user's query.
    Returns {"origin": str | None, "destination": str | None}.
    """
    extraction_prompt = f"""
أنت مساعد لاستخراج المعلومات. استخرج من الجملة التالية اسم المدينة أو المحافظة التي يريد المستخدم السفر منها (origin) 
واسم المدينة أو المحافظة التي يريد السفر إليها (destination).

أجب فقط بـ JSON بالشكل التالي بدون أي نص إضافي:
{{"origin": "اسم المكان أو null", "destination": "اسم المكان أو null"}}

الجملة: {query}
"""
    result = llm.invoke(extraction_prompt)
    text = result.content.strip()

    # strip markdown code fences if the model wraps its reply
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        entities = json.loads(text)
        origin = entities.get("origin")
        destination = entities.get("destination")
        # treat the string "null" as None
        if isinstance(origin, str) and origin.lower() in ("null", "none", ""):
            origin = None
        if isinstance(destination, str) and destination.lower() in ("null", "none", ""):
            destination = None
        return {"origin": origin, "destination": destination}
    except (json.JSONDecodeError, AttributeError):
        return {"origin": None, "destination": None}


# hybrid retriever with metadata filter + semantic fallback
def hybrid_retrieve(
    query: str,
    vecstore: PineconeVectorStore,
    llm: ChatGoogleGenerativeAI,
    pinecone_index,
    embeddings,
    k: int = 5,
) -> list:
    """
    Two-layer retrieval strategy:

    Layer 1 — Exact metadata match:
        Extract origin + destination with the LLM, then query Pinecone
        using a metadata filter on {city, destination}. If any direct
        routes are found, return ONLY those — semantic search is skipped.

    Layer 2 — Semantic similarity fallback:
        Used only when Layer 1 finds nothing (no entities extracted, or
        no exact match exists in the index). Returns top-k results by
        cosine similarity.
    """
    entities = extract_route_entities(query, llm)
    origin = entities.get("origin")
    destination = entities.get("destination")

    # Layer 1: exact metadata match
    if origin and destination:
        print(f"[Retrieval] Trying exact match: origin='{origin}' → destination='{destination}'")
        try:
            # Embed the natural-language route as the query vector, then let
            # the metadata filter do the exact matching work.
            #query_vector = embeddings.embed_query(f"سفر من {origin} إلى {destination}")
            query_vector = embeddings.embed_query(query)

            results = pinecone_index.query(
                vector=query_vector,
                top_k=20,           # fetch enough candidates so the filter has room to match
                include_metadata=True,
                filter={
                    "$and": [
                        {"$or": [{"city": {"$eq": origin}}, {"governorate": {"$eq": origin}}]},
                        {"destination": {"$eq": destination}},
                    ]
                },
            )

            matched_docs = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                content = (
                    f'من محافظة {meta.get("governorate", "")} مدينة {meta.get("city", "")} '
                    f'عبر محطة {meta.get("station", "")} يمكن السفر إلى '
                    f'{meta.get("destination", "")} بسعر {meta.get("price", "")} جنيه'
                )
                matched_docs.append(Document(page_content=content, metadata=meta))

            if matched_docs:
                print(f"[Retrieval] Layer 1: found {len(matched_docs)} direct route(s). Skipping semantic search.")
                return matched_docs

            print("[Retrieval] Layer 1: no direct route found, falling back to semantic search.")

        except Exception as e:
            print(f"[Retrieval] Layer 1 metadata filter failed ({e}), falling back to semantic search.")

    # Layer 2: semantic similarity fallback 
    print(f"[Retrieval] Layer 2: running semantic search with k={k}.")
    query_vector2 = embeddings.embed_query(query)
    return vecstore.similarity_search_by_vector(query_vector2, k=k)
    #return vecstore.similarity_search_by_vector(query, k=k)


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the travel chatbot API...")

    # Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in environment variables.")
    pc = Pinecone(api_key=pinecone_api_key)

    # Embeddings — must match the model used in pipeline.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Vector store (for semantic fallback) + raw index handle (for metadata filter)
    index_name = "chatbot-index"
    vecstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    pinecone_index = pc.Index(index_name)

    # LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=gemini_api_key,
    )

    # Prompt — explicitly instructs the LLM to prioritise direct routes
    prompt_template = """أنت مساعد ذكي متخصص في الاستعلام عن خطوط السفر بالحافلات.

قواعد مهمة يجب اتباعها بدقة:
- إذا وجدت خطاً مباشراً أو أكثر بين نقطة الانطلاق والوجهة في المعلومات المتاحة، اذكرهم أولاً وبوضوح.
- إذا لم يوجد خط مباشر في المعلومات، وضّح ذلك صراحةً , بعدها حاول الوصول لخطوط غير مباشرةأو تبديلات للوصول عبر محطات وسيطة.
- إذا كانت المعلومات المقدمة لا تحتوي على إجابة مباشرة، فحاول أن تستنتج أقرب معلومة.
- إذا لم تكن المعلومات كافية للإجابة، قل فقط: "عذرًا، لا أستطيع مساعدتك في هذا الاستفسار".

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}

الرد:"""
    PROMPT = PromptTemplate.from_template(prompt_template)

    # Attach everything to app state for use at request time
    app.state.llm = llm
    app.state.vecstore = vecstore
    app.state.pinecone_index = pinecone_index
    app.state.embeddings = embeddings
    app.state.prompt = PROMPT

    print("Travel chatbot API is ready to receive queries.")
    yield
    print("Shutting down the travel chatbot API.")


# app and endpoint definitions 
app = FastAPI(
    title="Travel Chatbot API",
    description="API for querying the travel chatbot",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def ask(request: QueryRequest):
    llm            = getattr(app.state, "llm", None)
    vecstore       = getattr(app.state, "vecstore", None)
    pinecone_index = getattr(app.state, "pinecone_index", None)
    embeddings     = getattr(app.state, "embeddings", None)
    PROMPT         = getattr(app.state, "prompt", None)

    if not all([llm, vecstore, pinecone_index, embeddings, PROMPT]):
        raise HTTPException(
            status_code=503,
            detail="النظام لم يكتمل تحميله بعد، حاول مرة أخرى.",
        )

    # hybrid retrieval
    docs = hybrid_retrieve(
        query=request.query,
        vecstore=vecstore,
        llm=llm,
        pinecone_index=pinecone_index,
        embeddings=embeddings,
        k=5,
    )

    # assemble context and generate answer
    context = "\n".join(f"- {doc.page_content}" for doc in docs[:5])  # limit context to top 5 results for brevity
    filled_prompt = PROMPT.format(context=context, question=request.query)
    result = llm.invoke(filled_prompt)
    answer = result.content.strip()

    return QueryResponse(response=answer)
