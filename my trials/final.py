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
# Each set is a group of equivalent names — every spelling/nickname that could
# appear in user queries OR in stored metadata. The $in filter matches ANY of
# them, so it doesn't matter which variant is stored vs. what the user typed.
# Add new groups as you discover mismatches in your data.
PLACE_VARIANT_GROUPS: list[set] = [
    {"الفيوم", "فيوم", "الفيّوم", "يوسف الصديق"},
    {"القاهرة", "القاهره", "كايرو"},
    {"الإسكندرية", "اسكندرية", "الاسكندرية", "إسكندرية", "اسكندريه"},
    # add more groups as needed …
]

def get_all_variants(name: str) -> list:
    """Return all known equivalent names for a place, including itself."""
    if not name:
        return [name]
    name = name.strip()
    for group in PLACE_VARIANT_GROUPS:
        if name in group:
            return list(group)
    return [name]


# ── intent + entity extraction ────────────────────────────────────────────────
def classify_and_extract(query: str, llm) -> dict:
    """
    Single LLM call that both classifies the user's intent AND extracts
    relevant entities (origin, destination, station, attribute).

    Intents:
      - route_search   : user wants to travel from A to B
      - price_inquiry  : user asks about ticket price
      - station_lookup : user asks what stations/routes exist in a place
      - general_info   : greeting, general question, out-of-scope
    """
    prompt = f"""
أنت مساعد ذكي لتحليل استفسارات السفر بالحافلات. حلل الجملة التالية واستخرج:

١. intent: نوع الاستفسار — اختر واحداً فقط من:
   - route_search   : المستخدم يريد السفر من مكان إلى آخر
   - price_inquiry  : المستخدم يسأل عن السعر أو التكلفة
   - station_lookup : المستخدم يسأل عن محطات أو خطوط في مكان معين
   - general_info   : سؤال عام أو تحية أو خارج النطاق

٢. origin      : مكان الانطلاق (مدينة أو محافظة أو محطة) — أو null
٣. destination : الوجهة (مدينة أو محافظة أو محطة) — أو null
٤. station     : اسم محطة محددة إذا ذُكرت — أو null
٥. attribute   : الخاصية المطلوبة مثل "سعر" أو "وقت" أو "محطات" — أو null

أجب فقط بـ JSON بالشكل التالي بدون أي نص إضافي:
{{
  "intent": "...",
  "origin": "... أو null",
  "destination": "... أو null",
  "station": "... أو null",
  "attribute": "... أو null"
}}

الجملة: {query}
"""
    result = llm.invoke(prompt)
    text = result.content.strip()

    # strip markdown fences if the model wraps the JSON
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    null_values = {"null", "none", ""}
    defaults = {
        "intent": "general_info",
        "origin": None, "destination": None,
        "station": None, "attribute": None,
    }
    try:
        parsed = json.loads(text)
        for key in defaults:
            val = parsed.get(key)
            if isinstance(val, str):
                val = val.strip()
                if val.lower() in null_values:
                    val = None
            parsed[key] = val
        # fill any missing keys with defaults
        for key, default in defaults.items():
            parsed.setdefault(key, default)
        return parsed
    except (json.JSONDecodeError, AttributeError):
        return defaults


# ── query rewriter ────────────────────────────────────────────────────────────
def rewrite_query_for_embedding(entities: dict) -> str:
    """
    Convert extracted entities into a formal Arabic sentence that closely
    matches the style of documents stored in Pinecone:
      "من محافظة X مدينة Y عبر محطة Z يمكن السفر إلى W بسعر N جنيه"

    Colloquial queries like "اركب ايه من القاهرة للفيوم؟" embed very
    differently from stored documents. Rewriting to a formal style closes
    that gap and dramatically improves cosine similarity scores.
    """
    origin      = entities.get("origin")
    destination = entities.get("destination")
    station     = entities.get("station")
    intent      = entities.get("intent", "route_search")

    if intent == "station_lookup" and origin and not destination:
        if station:
            return f"محطة {station} في {origin}"
        return f"خطوط السفر المتاحة من {origin}"

    if intent == "price_inquiry":
        if origin and destination:
            return f"سعر السفر من {origin} إلى {destination}"
        if destination:
            return f"سعر السفر إلى {destination}"

    # default: formal route sentence matching stored document style
    parts = []
    if origin:
        parts.append(f"من {origin}")
    if station:
        parts.append(f"عبر محطة {station}")
    if destination:
        parts.append(f"يمكن السفر إلى {destination}")
    if parts:
        return " ".join(parts)

    # absolute fallback: return original query
    return entities.get("_raw_query", "سفر")


# ── Pinecone helpers ──────────────────────────────────────────────────────────
def _meta_to_doc(meta: dict) -> Document:
    content = (
        f'من محافظة {meta.get("governorate", "")} مدينة {meta.get("city", "")} '
        f'عبر محطة {meta.get("station", "")} يمكن السفر إلى '
        f'{meta.get("destination", "")} بسعر {meta.get("price", "")} جنيه'
    )
    return Document(page_content=content, metadata=meta)


def _pinecone_filter_query(
    pinecone_index,
    query_vector: list,
    metadata_filter: dict,
    top_k: int = 50,
) -> list:
    results = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )
    return [
        _meta_to_doc(m["metadata"])
        for m in results.get("matches", [])
        if m.get("metadata")
    ]


def _deduplicate(docs: list) -> list:
    seen, unique = set(), []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


# ── hybrid retriever ──────────────────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    vecstore: PineconeVectorStore,
    llm,
    pinecone_index,
    embeddings,
    k: int = 5,
) -> list:
    """
    Intent-aware hybrid retrieval with four layers:

    Layer 1 — Exact multi-variant match (city/governorate + destination $in):
        Tries both city-level and governorate-level filters with all known
        spelling variants of each place. Uses a formally rewritten query
        vector instead of the raw user query for better cosine similarity.

    Layer 2 — Single-entity lookup (station_lookup / price_inquiry / one-sided):
        For queries with only origin OR only destination, or station-focused
        queries. Fetches all routes from/to the named place.

    Layer 3 — Smart indirect route construction:
        Finds routes FROM origin (any destination) and routes TO destination
        (any origin), then identifies intermediate cities that appear in both
        sets — i.e. cities you can reach from origin AND that have onward
        routes to destination. Only those connected pairs are returned,
        giving the LLM a clean path to reason about.

    Layer 4 — Rewritten semantic search (last resort):
        Uses the formally rewritten query (not the raw colloquial text) for
        embedding, which aligns much better with stored document style.
    """
    # ── classify intent and extract entities ──────────────────────────────────
    entities = classify_and_extract(query, llm)
    entities["_raw_query"] = query      # stash for fallback

    intent      = entities.get("intent", "general_info")
    origin      = entities.get("origin")
    destination = entities.get("destination")
    station     = entities.get("station")

    print(f"[Retrieval] intent='{intent}' origin='{origin}' destination='{destination}' station='{station}'")

    # ── short-circuit: general greeting or out-of-scope ───────────────────────
    if intent == "general_info" and not origin and not destination:
        print("[Retrieval] General/out-of-scope query — no retrieval needed.")
        return []

    # ── build rewritten query vector (better embedding alignment) ─────────────
    rewritten = rewrite_query_for_embedding(entities)
    print(f"[Retrieval] Rewritten query for embedding: '{rewritten}'")
    base_vector = embeddings.embed_query(rewritten)

    # ── Layer 1: exact match with variant expansion ───────────────────────────
    if origin and destination:
        origin_v = get_all_variants(origin)
        dest_v   = get_all_variants(destination)
        print(f"[Retrieval] Layer 1: origin_variants={origin_v}, dest_variants={dest_v}")

        # 1a: city-level match
        docs = _pinecone_filter_query(
            pinecone_index, base_vector,
            filter={"$and": [
                {"city":        {"$in": origin_v}},
                {"destination": {"$in": dest_v}},
            ]},
        )
        if docs:
            print(f"[Retrieval] Layer 1a hit: {len(docs)} route(s).")
            return _deduplicate(docs)

        # 1b: governorate-level match
        docs = _pinecone_filter_query(
            pinecone_index, base_vector,
            filter={"$and": [
                {"governorate": {"$in": origin_v}},
                {"destination": {"$in": dest_v}},
            ]},
        )
        if docs:
            print(f"[Retrieval] Layer 1b hit: {len(docs)} route(s).")
            return _deduplicate(docs)

        # 1c: station-level match (if a specific station was mentioned)
        if station:
            docs = _pinecone_filter_query(
                pinecone_index, base_vector,
                filter={"$and": [
                    {"station":     {"$eq": station}},
                    {"destination": {"$in": dest_v}},
                ]},
            )
            if docs:
                print(f"[Retrieval] Layer 1c (station) hit: {len(docs)} route(s).")
                return _deduplicate(docs)

    # ── Layer 2: single-entity lookup ─────────────────────────────────────────
    # Handles: station_lookup, price_inquiry with one side, or partial queries
    if (origin and not destination) or (destination and not origin) or intent == "station_lookup":
        place        = origin or destination
        place_v      = get_all_variants(place) if place else []
        is_origin    = bool(origin)

        print(f"[Retrieval] Layer 2: single-entity lookup for '{place}' (is_origin={is_origin})")

        if is_origin:
            docs = _pinecone_filter_query(
                pinecone_index, base_vector,
                filter={"$or": [
                    {"city":        {"$in": place_v}},
                    {"governorate": {"$in": place_v}},
                    {"station":     {"$in": place_v}},
                ]},
                top_k=30,
            )
        else:
            docs = _pinecone_filter_query(
                pinecone_index, base_vector,
                filter={"destination": {"$in": place_v}},
                top_k=30,
            )

        if docs:
            print(f"[Retrieval] Layer 2 hit: {len(docs)} route(s).")
            return _deduplicate(docs)

    # ── Layer 3: smart indirect route construction ────────────────────────────
    if origin and destination:
        origin_v = get_all_variants(origin)
        dest_v   = get_all_variants(destination)
        print(f"[Retrieval] Layer 3: smart indirect route construction.")

        # all destinations reachable from origin
        from_vec    = embeddings.embed_query(f"من {origin} يمكن السفر إلى")
        from_origin = _pinecone_filter_query(
            pinecone_index, from_vec,
            filter={"$or": [
                {"city":        {"$in": origin_v}},
                {"governorate": {"$in": origin_v}},
            ]},
            top_k=50,
        )

        # all origins that have routes TO destination
        to_vec         = embeddings.embed_query(f"يمكن السفر إلى {destination}")
        to_destination = _pinecone_filter_query(
            pinecone_index, to_vec,
            filter={"destination": {"$in": dest_v}},
            top_k=50,
        )

        if from_origin and to_destination:
            # find intermediate cities: appear as destination of leg-1
            # AND as city/governorate of leg-2
            reachable_from_origin = {
                doc.metadata.get("destination") for doc in from_origin
                if doc.metadata.get("destination")
            }
            intermediates_in_leg2 = {
                doc.metadata.get("city") for doc in to_destination
                if doc.metadata.get("city")
            } | {
                doc.metadata.get("governorate") for doc in to_destination
                if doc.metadata.get("governorate")
            }
            connecting_cities = reachable_from_origin & intermediates_in_leg2

            print(f"[Retrieval] Layer 3: {len(connecting_cities)} connecting city/cities found: {connecting_cities}")

            if connecting_cities:
                # keep only the legs that form valid connected pairs
                leg1 = [d for d in from_origin  if d.metadata.get("destination") in connecting_cities]
                leg2 = [d for d in to_destination if d.metadata.get("city") in connecting_cities
                                                   or d.metadata.get("governorate") in connecting_cities]
                connected = _deduplicate(leg1 + leg2)
                print(f"[Retrieval] Layer 3 hit: {len(leg1)} leg-1 + {len(leg2)} leg-2 = {len(connected)} docs.")
                return connected

            # no connecting city found but still return raw legs so LLM can try
            fallback = _deduplicate(from_origin + to_destination)
            print(f"[Retrieval] Layer 3 fallback: returning {len(fallback)} unconnected legs.")
            return fallback

    # ── Layer 4: rewritten semantic search ───────────────────────────────────
    print(f"[Retrieval] Layer 4: rewritten semantic search (k={k}).")
    return vecstore.similarity_search(rewritten, k=k)


# ── intent-specific prompts ───────────────────────────────────────────────────
PROMPTS = {
    "route_search": """أنت مساعد ذكي متخصص في خطوط السفر بالحافلات.

قواعد مهمة:
١. إذا وجدت خطوطاً مباشرة، اذكرها جميعاً بوضوح مع اسم المحطة والسعر.
٢. لا تقترح خطوطاً غير مباشرة إذا وُجد خط مباشر.
٣. إذا لم يوجد خط مباشر، وضّح ذلك، ثم اقترح مساراً عبر محطة وسيطة موجودة في المعلومات مع ذكر التكلفة الإجمالية.
٤. إذا لم تكن المعلومات كافية، قل: "عذرًا، لا تتوفر رحلات مباشرة أو غير مباشرة لهذا المسار في بياناتنا حالياً."

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}
الرد:""",

    "price_inquiry": """أنت مساعد ذكي متخصص في خطوط السفر بالحافلات.
أجب بوضوح عن سعر التذكرة أو نطاق الأسعار المتاح بين المحطات المذكورة.
إذا كان هناك أكثر من خيار، اذكر أرخص وأغلى سعر والمحطة المقابلة لكل منهما.

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}
الرد:""",

    "station_lookup": """أنت مساعد ذكي متخصص في خطوط السفر بالحافلات.
اعرض جميع المحطات أو الخطوط المتاحة من/إلى المكان المذكور بشكل منظم.
اذكر اسم المحطة والوجهة والسعر لكل خط.

المعلومات المتاحة:
{context}

سؤال المستخدم: {question}
الرد:""",

    "general_info": """أنت مساعد ذكي ومفيد لخدمة ركاب الحافلات.
أجب بلطف على سؤال المستخدم. إذا كان السؤال خارج نطاق خطوط السفر بالحافلات،
وضّح ذلك بأدب واقترح ما يمكنك المساعدة فيه.

المعلومات المتاحة (قد تكون فارغة):
{context}

سؤال المستخدم: {question}
الرد:""",
}


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

    # classify + extract in one call
    entities = classify_and_extract(request.query, llm)
    intent   = entities.get("intent", "general_info")

    # retrieve relevant documents
    docs = hybrid_retrieve(
        query=request.query,
        vecstore=vecstore,
        llm=llm,
        pinecone_index=pinecone_index,
        embeddings=embeddings,
        k=5,
    )

    # select the right prompt for this intent
    prompt_template = PROMPTS.get(intent, PROMPTS["general_info"])
    PROMPT          = PromptTemplate.from_template(prompt_template)

    context       = "\n".join(f"- {doc.page_content}" for doc in docs) if docs else "لا توجد معلومات متاحة."
    filled_prompt = PROMPT.format(context=context, question=request.query)
    result        = llm.invoke(filled_prompt)
    answer        = result.content.strip()

    return QueryResponse(response=answer)
