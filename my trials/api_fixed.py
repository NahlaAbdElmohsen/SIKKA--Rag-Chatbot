# importing libraries
import os
import re
import json
from contextlib import asynccontextmanager
from collections import defaultdict

import gspread
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

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


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str


# ─────────────────────────────────────────────────────────────────────────────
# Google Sheets — live price source
#
# Setup steps:
#   1. Create a Google Cloud service account and download its JSON key.
#   2. Share your Google Sheet with the service account email (viewer is enough).
#   3. Set these env vars in your .env:
#        GOOGLE_SHEETS_CREDENTIALS_JSON = /path/to/service_account.json
#        GOOGLE_SHEETS_SPREADSHEET_ID   = <your spreadsheet id from the URL>
#        GOOGLE_SHEETS_WORKSHEET_NAME   = Sheet1   (or whatever tab name you use)
#   4. Your sheet must have these column headers (same names as the Excel):
#        governate | city | station_name | destination | price
# ─────────────────────────────────────────────────────────────────────────────

def load_data_from_sheets() -> pd.DataFrame:
    """
    Fetch the route/price data from Google Sheets and return a DataFrame.
    Prices are read live every time the app starts (or when manually refreshed).
    """
    creds_path      = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
    spreadsheet_id  = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
    worksheet_name  = os.getenv("GOOGLE_SHEETS_WORKSHEET_NAME", "Sheet1")

    if not creds_path or not spreadsheet_id:
        raise RuntimeError(
            "Missing env vars: GOOGLE_SHEETS_CREDENTIALS_JSON and/or "
            "GOOGLE_SHEETS_SPREADSHEET_ID must be set."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds  = Credentials.from_service_account_file(creds_path, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)
    records = sheet.get_all_records()           # list of dicts keyed by header row
    df = pd.DataFrame(records)

    # strip whitespace on all string columns
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    # ensure price is numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)

    print(f"[Sheets] Loaded {len(df)} rows from Google Sheets.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Destination parser  (mirrors pipeline.py — must stay in sync)
# ─────────────────────────────────────────────────────────────────────────────

def parse_destination(dest: str):
    """
    Split "الإسكندرية (مكيف)"  →  ("الإسكندرية", "مكيف")
    Split "بورتو السخنة"        →  ("بورتو السخنة", "")
    """
    dest  = dest.strip()
    match = re.search(r'\((.+?)\)', dest)
    line_type = match.group(1) if match else ""
    city      = re.sub(r'\s*\(.+?\)', '', dest).strip()
    return city, line_type


# ─────────────────────────────────────────────────────────────────────────────
# Route graph  — built once from Google Sheets data at startup
# ─────────────────────────────────────────────────────────────────────────────

def build_route_graph(df: pd.DataFrame) -> dict:
    """
    Build adjacency map:
        graph[origin_city] = [
            {destination, line_type, price, station, governorate}, ...
        ]

    Prices come from Google Sheets — NOT from Pinecone.
    Pinecone is used only for semantic search (route discovery).
    """
    graph = defaultdict(list)
    for _, row in df.iterrows():
        city_dest, line_type = parse_destination(str(row["destination"]))
        graph[row["city"]].append({
            "destination": city_dest,
            "line_type":   line_type,
            "price":       row["price"],       # live price from Sheets
            "station":     row["station_name"],
            "governorate": row["governate"],
        })
    return graph


def find_routes(graph: dict, origin: str, destination: str) -> dict:
    """
    2-hop BFS — returns direct routes and routes with exactly one transfer.

    Covers ~99% of real-world regional bus network queries.
    The LLM never guesses routes; Python always decides them deterministically.
    """
    results = {"direct": [], "one_transfer": []}

    # Direct legs: origin → destination
    for route in graph.get(origin, []):
        if route["destination"] == destination:
            results["direct"].append(route)

    # 1-transfer legs: origin → hub → destination
    for first_leg in graph.get(origin, []):
        hub = first_leg["destination"]
        if hub == destination:
            continue            # already captured as direct
        for second_leg in graph.get(hub, []):
            if second_leg["destination"] == destination:
                results["one_transfer"].append({
                    "via":         hub,
                    "first_leg":   first_leg,
                    "second_leg":  second_leg,
                    "total_price": first_leg["price"] + second_leg["price"],
                })

    return results


def format_routes_as_context(origin: str, destination: str, routes: dict) -> str:
    """
    Convert routing results into Arabic text injected into the LLM prompt.
    Direct routes are always listed first and marked as PRIMARY.
    Indirect routes are marked as ALTERNATIVE so the LLM knows the priority.
    """
    lines = []

    # ── Direct routes — always first, marked PRIMARY ──────────────────────────
    if routes["direct"]:
        lines.append(f"[PRIMARY] خط مباشر من {origin} إلى {destination}:")
        for r in routes["direct"]:
            lt = f" ({r['line_type']})" if r["line_type"] else ""
            lines.append(
                f"  • {r['station']} ← {destination}{lt} — السعر: {r['price']} جنيه"
            )

    # ── Indirect routes — always after direct, marked ALTERNATIVE ────────────
    if routes["one_transfer"]:
        # Only label as alternative if a direct route also exists
        label = "[ALTERNATIVE]" if routes["direct"] else "[PRIMARY - لا يوجد خط مباشر]"
        lines.append(f"\n{label} خطوط بتحويلة واحدة من {origin} إلى {destination}:")
        for r in routes["one_transfer"]:
            f1, f2 = r["first_leg"], r["second_leg"]
            lt2 = f" ({f2['line_type']})" if f2["line_type"] else ""
            lines.append(
                f"  • {origin} ({f1['station']}) ← {r['via']} — {f1['price']} جنيه"
                f"  ثم {r['via']} ({f2['station']}) ← {destination}{lt2} — {f2['price']} جنيه"
                f"  | الإجمالي: {r['total_price']} جنيه"
            )

    if not lines:
        lines.append(
            f"لا توجد خطوط مباشرة أو غير مباشرة من {origin} إلى {destination} "
            f"في قاعدة البيانات."
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction  — LLM extracts origin / destination from free Arabic text
# ─────────────────────────────────────────────────────────────────────────────

def extract_route_entities(query: str, llm: ChatGoogleGenerativeAI) -> dict:
    """
    Ask the LLM to extract origin and destination from the user's query.
    Returns {"origin": str | None, "destination": str | None}.
    """
    extraction_prompt = f"""
أنت مساعد لاستخراج المعلومات. استخرج من الجملة التالية اسم المدينة أو المحافظة 
التي يريد المستخدم السفر منها (origin) واسم المدينة أو المحافظة التي يريد السفر إليها (destination).

أجب فقط بـ JSON بالشكل التالي بدون أي نص إضافي:
{{"origin": "اسم المكان أو null", "destination": "اسم المكان أو null"}}

الجملة: {query}
"""
    result = llm.invoke(extraction_prompt)
    text   = result.content.strip()

    # strip markdown code fences if the model wraps its reply
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        entities    = json.loads(text)
        origin      = entities.get("origin")
        destination = entities.get("destination")
        # treat the string "null" as None
        if isinstance(origin, str) and origin.lower() in ("null", "none", ""):
            origin = None
        if isinstance(destination, str) and destination.lower() in ("null", "none", ""):
            destination = None
        return {"origin": origin, "destination": destination}
    except (json.JSONDecodeError, AttributeError):
        return {"origin": None, "destination": None}


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval
#
# Priority order:
#   1. Route graph  — if both origin & destination are extracted, run 2-hop BFS.
#                     Prices are always current (from Google Sheets).
#   2. Pinecone metadata filter — if graph returns nothing (e.g. entity mismatch),
#                     fall back to exact metadata search in Pinecone (no price).
#   3. Semantic similarity — final fallback for open-ended / vague queries.
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query:          str,
    vecstore:       PineconeVectorStore,
    llm:            ChatGoogleGenerativeAI,
    pinecone_index,
    embeddings,
    route_graph:    dict,
    k:              int = 5,
) -> tuple[list[Document], str]:
    """
    Returns (docs, routing_context).

    routing_context is a pre-formatted Arabic string from the route graph.
    If it is non-empty, it takes priority in the LLM prompt over doc content.
    """
    entities    = extract_route_entities(query, llm)
    origin      = entities.get("origin")
    destination = entities.get("destination")

    routing_context = ""

    # ── Layer 1: Route graph (deterministic, live prices) ────────────────────
    if origin and destination:
        print(f"[Routing] Graph lookup: '{origin}' → '{destination}'")
        routes = find_routes(route_graph, origin, destination)
        if routes["direct"] or routes["one_transfer"]:
            routing_context = format_routes_as_context(origin, destination, routes)
            print(f"[Routing] Graph found routes — returning graph context only.")
            # Return empty docs; routing_context carries all the info needed.
            return [], routing_context

        print("[Routing] Graph: no routes found, trying Pinecone metadata filter.")

        # ── Layer 2: Pinecone metadata filter (structure match, no live price) ─
        try:
            query_vector = embeddings.embed_query(query)
            results = pinecone_index.query(
                vector=query_vector,
                top_k=20,
                include_metadata=True,
                filter={
                    "$and": [
                        {"city":        {"$eq": origin}},
                        {"destination": {"$eq": destination}},
                    ]
                },
            )
            matched_docs = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                # Note: price in Pinecone metadata may be stale —
                # we flag this in the content so the LLM is aware.
                lt      = f" ({meta.get('line_type', '')})" if meta.get("line_type") else ""
                content = (
                    f'من محافظة {meta.get("governorate", "")} '
                    f'مدينة {meta.get("city", "")} '
                    f'عبر محطة {meta.get("station", "")} '
                    f'يمكن السفر إلى {meta.get("destination", "")}{lt} '
                    f'(السعر قد يكون غير محدّث، يُرجى التحقق)'
                )
                matched_docs.append(Document(page_content=content, metadata=meta))

            if matched_docs:
                print(f"[Retrieval] Layer 2: {len(matched_docs)} doc(s) from metadata filter.")
                return matched_docs, routing_context

            print("[Retrieval] Layer 2: no metadata match, falling back to semantic search.")

        except Exception as e:
            print(f"[Retrieval] Layer 2 failed ({e}), falling back to semantic search.")

    # ── Layer 3: Semantic similarity fallback ────────────────────────────────
    print(f"[Retrieval] Layer 3: semantic search k={k}.")
    query_vector = embeddings.embed_query(query)
    docs = vecstore.similarity_search_by_vector(query_vector, k=k)
    return docs, routing_context


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the travel chatbot API...")

    # ── Google Sheets → DataFrame → Route graph ───────────────────────────────
    df          = load_data_from_sheets()
    route_graph = build_route_graph(df)
    print(f"[Startup] Route graph built: {len(route_graph)} origin cities.")

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")
    pc = Pinecone(api_key=pinecone_api_key)

    # ── Embeddings (must match model used in pipeline.py) ────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    index_name     = "chatbot-index"
    vecstore       = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    pinecone_index = pc.Index(index_name)

    # ── LLM ───────────────────────────────────────────────────────────────────
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        api_key=gemini_api_key,
    )

    # ── Prompt ────────────────────────────────────────────────────────────────
    # The prompt receives either:
    #   • routing_context  — pre-computed Arabic route summary (authoritative, live prices)
    #   • doc_context      — semantic / metadata retrieval results (fallback)
    # The LLM must present the routing_context first when available.
    prompt_template = """أنت مساعد ذكي متخصص في الاستعلام عن خطوط السفر بالحافلات.

قواعد صارمة يجب اتباعها بالترتيب:

1. ابدأ دائماً بالخطوط المُعلَّمة [PRIMARY] في المعلومات أدناه — هذه هي إجابتك الرئيسية.
2. إذا كان [PRIMARY] خطاً مباشراً، اذكره بوضوح مع محطة الانطلاق والسعر.
3. فقط بعد ذكر الخط المباشر، يمكنك ذكر الخطوط المُعلَّمة [ALTERNATIVE] كخيارات إضافية للمستخدم — لا تبدأ بها أبداً.
4. إذا كان [PRIMARY] خطاً بتحويلة (لا يوجد مباشر)، اذكره مع تفصيل كل رحلة وسعرها والإجمالي.
5. لا تخترع مسارات أو أسعار خارج المعلومات المقدمة.
6. إذا لم تكن المعلومات كافية، قل فقط: "عذرًا، لا أستطيع مساعدتك في هذا الاستفسار".

--- نتائج البحث عن المسار ---
{routing_context}

--- معلومات إضافية ---
{doc_context}

سؤال المستخدم: {question}

الرد:"""
    PROMPT = PromptTemplate.from_template(prompt_template)

    # ── Attach to app state ───────────────────────────────────────────────────
    app.state.llm           = llm
    app.state.vecstore      = vecstore
    app.state.pinecone_index = pinecone_index
    app.state.embeddings    = embeddings
    app.state.prompt        = PROMPT
    app.state.route_graph   = route_graph

    print("Travel chatbot API is ready.")
    yield
    print("Shutting down the travel chatbot API.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Travel Chatbot API",
    description="Bus route chatbot with live Google Sheets prices and graph routing.",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def ask(request: QueryRequest):
    llm            = getattr(app.state, "llm",            None)
    vecstore       = getattr(app.state, "vecstore",       None)
    pinecone_index = getattr(app.state, "pinecone_index", None)
    embeddings     = getattr(app.state, "embeddings",     None)
    PROMPT         = getattr(app.state, "prompt",         None)
    route_graph    = getattr(app.state, "route_graph",    None)

    if not all([llm, vecstore, pinecone_index, embeddings, PROMPT, route_graph]):
        raise HTTPException(
            status_code=503,
            detail="النظام لم يكتمل تحميله بعد، حاول مرة أخرى.",
        )

    # Hybrid retrieval — graph first, then Pinecone fallbacks
    docs, routing_context = hybrid_retrieve(
        query=          request.query,
        vecstore=       vecstore,
        llm=            llm,
        pinecone_index= pinecone_index,
        embeddings=     embeddings,
        route_graph=    route_graph,
        k=5,
    )

    # Build prompt context
    doc_context = "\n".join(f"- {doc.page_content}" for doc in docs[:5]) if docs else "لا توجد معلومات إضافية."

    filled_prompt = PROMPT.format(
        routing_context = routing_context or "لا توجد نتائج مسار محددة.",
        doc_context     = doc_context,
        question        = request.query,
    )

    result = llm.invoke(filled_prompt)
    answer = result.content.strip()

    return QueryResponse(response=answer)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: reload prices endpoint
# Call POST /reload-prices to refresh the route graph from Google Sheets
# without restarting the server — useful when you update prices in the sheet.
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reload-prices", include_in_schema=True)
async def reload_prices():
    """
    Reload the route graph from Google Sheets.
    Use this after updating prices in the sheet — no server restart needed.
    """
    try:
        df = load_data_from_sheets()
        app.state.route_graph = build_route_graph(df)
        return {"status": "ok", "message": f"تم تحديث الأسعار. عدد المسارات: {len(app.state.route_graph)} مدينة انطلاق."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل تحديث الأسعار: {e}")
