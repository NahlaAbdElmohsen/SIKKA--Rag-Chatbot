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
from langchain_core.prompts import PromptTemplate
from warnings import filterwarnings
filterwarnings("ignore")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()


# ── request / response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str


# ── place name variant groups ─────────────────────────────────────────────────
# Each set groups every spelling/nickname for the same place — both what users
# type AND what may be stored in metadata. $in filters match any of them.
PLACE_VARIANT_GROUPS: list[set] = [
    {"الفيوم", "فيوم", "الفيّوم", "يوسف الصديق"},
    {"القاهرة", "القاهره", "كايرو"},
    {"الإسكندرية", "اسكندرية", "الاسكندرية", "إسكندرية", "اسكندريه"},
    # add more groups as needed …
]

def get_all_variants(name: str) -> list:
    """Return every known spelling variant for a place name, including itself."""
    if not name:
        return [name]
    name = name.strip()
    for group in PLACE_VARIANT_GROUPS:
        if name in group:
            return list(group)
    return [name]


# ── entity extraction ─────────────────────────────────────────────────────────
def extract_route_entities(query: str, llm) -> dict:
    """Extract origin and destination from a free-text Arabic query."""
    extraction_prompt = f"""
أنت مساعد لاستخراج المعلومات. استخرج من الجملة التالية اسم المدينة أو المحافظة التي يريد المستخدم السفر منها (origin) 
واسم المدينة أو المحافظة التي يريد السفر إليها (destination).

أجب فقط بـ JSON بالشكل التالي بدون أي نص إضافي:
{{"origin": "اسم المكان أو null", "destination": "اسم المكان أو null"}}

الجملة: {query}
"""
    result = llm.invoke(extraction_prompt)
    text = result.content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        entities = json.loads(text)
        origin      = entities.get("origin")
        destination = entities.get("destination")
        null_values = {"null", "none", ""}
        if isinstance(origin,      str) and origin.lower()      in null_values: origin      = None
        if isinstance(destination, str) and destination.lower() in null_values: destination = None
        if origin:      origin      = origin.strip()
        if destination: destination = destination.strip()
        return {"origin": origin, "destination": destination}
    except (json.JSONDecodeError, AttributeError):
        return {"origin": None, "destination": None}


# ── query rewriter ────────────────────────────────────────────────────────────
def rewrite_for_embedding(origin: str = None, destination: str = None) -> str:
    """
    Build a formal Arabic sentence that mirrors the style of stored documents:
      "من محافظة X مدينة Y عبر محطة Z يمكن السفر إلى W بسعر N جنيه"
    Colloquial queries embed very differently from stored docs.
    Rewriting closes that vector-space gap and improves cosine similarity.
    """
    parts = []
    if origin:      parts.append(f"من {origin}")
    if destination: parts.append(f"يمكن السفر إلى {destination}")
    return " ".join(parts) if parts else "سفر"


# ── Pinecone helpers ──────────────────────────────────────────────────────────
def _meta_to_doc(meta: dict) -> Document:
    content = (
        f'من محافظة {meta.get("governorate", "")} مدينة {meta.get("city", "")} '
        f'عبر محطة {meta.get("station", "")} يمكن السفر إلى '
        f'{meta.get("destination", "")} بسعر {meta.get("price", "")} جنيه'
    )
    return Document(page_content=content, metadata=meta)


def _filter_query(pinecone_index, vector: list, metadata_filter: dict, top_k: int = 50) -> list:
    results = pinecone_index.query(
        vector=vector, top_k=top_k, include_metadata=True, filter=metadata_filter
    )
    return [_meta_to_doc(m["metadata"]) for m in results.get("matches", []) if m.get("metadata")]


def _dedupe(docs: list) -> list:
    seen, out = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            out.append(d)
    return out


# ── indirect route finder ─────────────────────────────────────────────────────
def find_indirect_routes(
    origin: str,
    destination: str,
    origin_variants: list,
    dest_variants: list,
    pinecone_index,
    embeddings,
) -> tuple[list, list, set]:
    """
    Finds all valid indirect routes between origin and destination.

    Step 1 — Fetch leg-1: all routes departing FROM origin (to any city).
    Step 2 — Fetch leg-2: all routes arriving AT destination (from any city).
    Step 3 — Find connecting cities: cities that appear as a *destination*
             in leg-1 AND as a *city/governorate* in leg-2. These are the
             valid intermediate stops.
    Step 4 — Filter both legs to only keep pairs that form a real connected
             path through one of those intermediate cities.

    Returns (leg1_docs, leg2_docs, connecting_cities).
    If connecting_cities is empty, both raw leg lists are still returned so
    the LLM can attempt its own reasoning.
    """
    # ── leg 1: all routes departing from origin ───────────────────────────────
    vec_from = embeddings.embed_query(rewrite_for_embedding(origin=origin))
    leg1 = _filter_query(
        pinecone_index, vec_from,
        filter={"$or": [
            {"city":        {"$in": origin_variants}},
            {"governorate": {"$in": origin_variants}},
        ]},
        top_k=50,
    )

    # ── leg 2: all routes arriving at destination ─────────────────────────────
    vec_to = embeddings.embed_query(rewrite_for_embedding(destination=destination))
    leg2 = _filter_query(
        pinecone_index, vec_to,
        filter={"destination": {"$in": dest_variants}},
        top_k=50,
    )

    if not leg1 or not leg2:
        print(f"[Indirect] leg1={len(leg1)} docs, leg2={len(leg2)} docs — one side empty.")
        return leg1, leg2, set()

    # ── find intermediate cities (intersection of leg-1 destinations and leg-2 origins)
    # leg-1 destinations = cities you can reach FROM origin
    reachable = {d.metadata.get("destination") for d in leg1 if d.metadata.get("destination")}

    # leg-2 origins = cities that HAVE a route TO destination
    leg2_cities = (
        {d.metadata.get("city")        for d in leg2 if d.metadata.get("city")}
      | {d.metadata.get("governorate") for d in leg2 if d.metadata.get("governorate")}
    )

    connecting_cities = reachable & leg2_cities
    print(f"[Indirect] Reachable from origin: {len(reachable)} cities.")
    print(f"[Indirect] Cities with route to dest: {len(leg2_cities)} cities.")
    print(f"[Indirect] Connecting cities: {connecting_cities}")

    if not connecting_cities:
        # no proven connection found — return raw legs anyway for LLM to try
        return _dedupe(leg1), _dedupe(leg2), set()

    # ── filter to only legs that form a valid connected pair ──────────────────
    leg1_filtered = [d for d in leg1 if d.metadata.get("destination") in connecting_cities]
    leg2_filtered = [d for d in leg2 if d.metadata.get("city")        in connecting_cities
                                     or d.metadata.get("governorate") in connecting_cities]

    print(f"[Indirect] Filtered: {len(leg1_filtered)} leg-1 + {len(leg2_filtered)} leg-2 docs.")
    return _dedupe(leg1_filtered), _dedupe(leg2_filtered), connecting_cities


# ── hybrid retriever ──────────────────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    vecstore: PineconeVectorStore,
    llm,
    pinecone_index,
    embeddings,
    k: int = 5,
) -> tuple[list, str]:
    """
    Returns (docs, route_type) where route_type is one of:
      "direct"   — Layer 1 found direct route(s)
      "indirect" — Layer 2 found connected indirect legs
      "semantic" — fell back to semantic search
      "none"     — nothing found

    Layer 1 — Exact multi-variant direct match:
        Uses $in filters with all spelling variants of origin and destination
        against both city-level and governorate-level metadata fields.
        Uses a rewritten formal query vector for better cosine alignment.

    Layer 2 — Indirect route reasoning:
        Splits retrieval into leg-1 (from origin) and leg-2 (to destination),
        finds intermediate connecting cities, and returns only the doc pairs
        that form real connected paths. The LLM is told explicitly it is
        working with indirect legs, not direct routes.

    Layer 3 — Rewritten semantic search (last resort):
        Uses the formally rewritten query (not raw colloquial text) so the
        embedding aligns with stored document style.
    """
    entities    = extract_route_entities(query, llm)
    origin      = entities.get("origin")
    destination = entities.get("destination")

    print(f"[Retrieval] origin='{origin}' destination='{destination}'")

    if origin and destination:
        origin_v = get_all_variants(origin)
        dest_v   = get_all_variants(destination)
        base_vec = embeddings.embed_query(rewrite_for_embedding(origin, destination))

        # ── Layer 1a: city-level direct match ────────────────────────────────
        docs = _filter_query(pinecone_index, base_vec, filter={"$and": [
            {"city":        {"$in": origin_v}},
            {"destination": {"$in": dest_v}},
        ]})
        if docs:
            print(f"[Retrieval] Layer 1a direct hit: {len(docs)} route(s).")
            return _dedupe(docs), "direct"

        # ── Layer 1b: governorate-level direct match ──────────────────────────
        docs = _filter_query(pinecone_index, base_vec, filter={"$and": [
            {"governorate": {"$in": origin_v}},
            {"destination": {"$in": dest_v}},
        ]})
        if docs:
            print(f"[Retrieval] Layer 1b direct hit: {len(docs)} route(s).")
            return _dedupe(docs), "direct"

        # ── Layer 2: indirect route reasoning ────────────────────────────────
        print(f"[Retrieval] No direct route. Searching for indirect legs…")
        leg1, leg2, connecting = find_indirect_routes(
            origin, destination, origin_v, dest_v, pinecone_index, embeddings
        )
        if leg1 or leg2:
            all_legs = _dedupe(leg1 + leg2)
            route_type = "indirect" if connecting else "indirect_unconfirmed"
            print(f"[Retrieval] Layer 2 indirect: {len(all_legs)} leg docs, connecting={connecting}.")
            return all_legs, route_type

    # ── Layer 3: rewritten semantic search ───────────────────────────────────
    rewritten = rewrite_for_embedding(origin, destination)
    print(f"[Retrieval] Layer 3 semantic search: '{rewritten}'")
    results = vecstore.similarity_search(rewritten, k=k)
    return results, "semantic" if results else "none"


# ── intent-aware prompts ──────────────────────────────────────────────────────
PROMPT_DIRECT = PromptTemplate.from_template("""أنت مساعد ذكي متخصص في خطوط السفر بالحافلات.

وجدنا خطوطاً مباشرة بين نقطة الانطلاق والوجهة. اعرضها جميعاً بوضوح مع اسم المحطة والسعر.

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}
الرد:""")

PROMPT_INDIRECT = PromptTemplate.from_template("""أنت مساعد ذكي متخصص في خطوط السفر بالحافلات.

لا يوجد خط مباشر بين نقطة الانطلاق والوجهة. المعلومات التالية تحتوي على **رحلتين منفصلتين** يمكن تركيبهما معاً:
- الرحلة الأولى (leg 1): من نقطة الانطلاق إلى مدينة وسيطة.
- الرحلة الثانية (leg 2): من تلك المدينة الوسيطة إلى الوجهة النهائية.

مهمتك:
١. حدد المدينة الوسيطة المشتركة بين الرحلتين.
٢. اذكر تفاصيل كل رحلة (المحطة، الوجهة، السعر).
٣. احسب التكلفة الإجمالية التقديرية لكامل الرحلة.
٤. إذا لم تجد مدينة وسيطة مشتركة واضحة، اذكر الخيارات المتاحة وقل للمستخدم إنه سيحتاج للتأكيد.

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}
الرد:""")

PROMPT_SEMANTIC = PromptTemplate.from_template("""أنت مساعد ذكي متخصص في خطوط السفر بالحافلات.

استخدم المعلومات التالية للإجابة على سؤال المستخدم بأفضل ما يمكن.
إذا لم تكن المعلومات كافية، قل: "عذرًا، لا أستطيع مساعدتك في هذا الاستفسار حالياً."

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}
الرد:""")


# ── lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the travel chatbot API...")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in environment variables.")
    pc = Pinecone(api_key=pinecone_api_key)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    index_name     = "chatbot-index"
    vecstore       = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    pinecone_index = pc.Index(index_name)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=gemini_api_key,
    )

    app.state.llm            = llm
    app.state.vecstore       = vecstore
    app.state.pinecone_index = pinecone_index
    app.state.embeddings     = embeddings

    print("Travel chatbot API is ready to receive queries.")
    yield
    print("Shutting down the travel chatbot API.")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Travel Chatbot API",
    description="API for querying the travel chatbot",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def ask(request: QueryRequest):
    llm            = getattr(app.state, "llm",            None)
    vecstore       = getattr(app.state, "vecstore",       None)
    pinecone_index = getattr(app.state, "pinecone_index", None)
    embeddings     = getattr(app.state, "embeddings",     None)

    if not all([llm, vecstore, pinecone_index, embeddings]):
        raise HTTPException(status_code=503, detail="النظام لم يكتمل تحميله بعد، حاول مرة أخرى.")

    docs, route_type = hybrid_retrieve(
        query=request.query,
        vecstore=vecstore,
        llm=llm,
        pinecone_index=pinecone_index,
        embeddings=embeddings,
        k=5,
    )
    print(f"[API] route_type='{route_type}', docs={len(docs)}")

    # pick the prompt that matches what kind of data we're sending the LLM
    if route_type == "direct":
        PROMPT = PROMPT_DIRECT
    elif route_type in ("indirect", "indirect_unconfirmed"):
        PROMPT = PROMPT_INDIRECT
    else:
        PROMPT = PROMPT_SEMANTIC

    context       = "\n".join(f"- {doc.page_content}" for doc in docs) if docs else "لا توجد معلومات متاحة."
    filled_prompt = PROMPT.format(context=context, question=request.query)
    result        = llm.invoke(filled_prompt)
    answer        = result.content.strip()

    return QueryResponse(response=answer)
