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
from collections import defaultdict
import pandas as pd
import re
load_dotenv()


# request / response models 
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
# updates are here
# CITY_VARIANTS = [{'الإسكندرية', 'اسكندرية', 'اسكندريه', 'الاسكندرية'},
#                  {'المحلة الكبرى', 'المحلة',"المحله"},
#                  {'بورسعيد', 'بور سعيد'},
#                  {'السخنة', "العين السخنة"},
#                  {'رأس سدر', 'راس سدر'},
#                  {'ديرب نجم', 'ديرب'},
#                  {"العاشر من رمضان", "العاشر"},
#                  {"رأس البر", "راس البر"},
#                  {"أبو المطامير","ابو المطامير"},
#                  {"المنصورة", "المنصوره"},
#                  {"إيتاى البارود","ايتاى البارود"},
#                  {"مدينة بدر", "بدر"},
#                  {"6 اكتوبر","6 أكتوبر","اكتوبر","أكتوبر","السادس من أكتوبر","السادس من اكتوبر"},
#                  {"التجمع الخامس","التجمع"},
#                  {"العبور", "مدينة العبور"},
#                  {"الإسماعيلية", "الاسماعيلية", "اسماعيلية"},
#                  {"عبود","موقف عبود"},
#                  {"المرج","موقف المرج","مدينة المرج"},
#                  {"السلام","موقف السلام","مدينة السلام"},
#                  {"إهناسيا","اهناسيا"},
#                  {"أجا","اجا"},
#                  {"أولاد صقر","ولاد صقر"},
#                  {"العاصمة الإدارية","العاصمة الجديدة","العاصمة الادارية"},
#                  {"الزاوية الحمراء","الزاوية الحمرا"},
#                  {"الزاوية الخضراء","الزاوية الخضرا"},
#                  {"القاهرة","القاهره"},
#                  {"الجيزة","الجيزه"},
#                  {"جمصة","جمصه"}]

# def get_all_variants(name: str) -> list:
#     """
#     Return all variants of a city name.
#     If the name is not recognized, return it as-is.
#     """
#     name = name.strip()
#     for variants in CITY_VARIANTS:
#         if name in variants:
#             return list(variants)  # return all variants as a list
#     return [name]  # if no match, return original as a list

def parse_destination(dest: str):
    """
    Split a destination string into (city_name, line_type).

    Examples:
        "الإسكندرية (مكيف)"  ->  ("الإسكندرية", "مكيف")
        "السويس (بيجو)"      ->  ("السويس",      "بيجو")
        "بورتو السخنة"       ->  ("بورتو السخنة", "")
        "رأس سدر"            ->  ("رأس سدر",      "")
    """
    dest = dest.strip()
    match = re.search(r'\((.+?)\)', dest)          # extract text inside ( )
    line_type = match.group(1) if match else ""    # e.g. "مكيف", "بيجو", or ""
    city = re.sub(r'\s*\(.+?\)', '', dest).strip() # remove the ( ) part entirely
    return city, line_type

def build_route_graph(df: pd.DataFrame ) -> dict:
    """
    Build adjacency map:
        graph[origin_city] = [
            {destination, line_type, price, station, governorate}, ...
        ]
    Pinecone is used only for semantic search (route discovery).
    """
    graph = defaultdict(list)
    for _, row in df.iterrows():
        city_dest, line_type = parse_destination(row["destination"])
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
    results = []
 
    # 1-transfer legs: origin → hub → destination
    for first_leg in graph.get(origin, []):
        hub = first_leg["destination"]
        if hub == destination:
            continue            # already captured as direct
        for second_leg in graph.get(hub, []):
            if second_leg["destination"] == destination:
                results.append({
                    "via":         hub,
                    "first_leg":   first_leg,
                    "second_leg":  second_leg,
                    "total_price": first_leg["price"] + second_leg["price"],
                })
 
    return results

def routes_to_documents(indirect_routes: dict) -> list:
    """
    Converts the output of find_routes() into a list of LangChain Document objects.
    find_routes() returns raw dicts — this builds readable Arabic content strings
    and attaches the dict itself as metadata.
    """
    docs = []
 
    for route in indirect_routes:
        first  = route["first_leg"]
        second = route["second_leg"]
        content = (
            f'لا يوجد خط مباشر، لكن يمكن السفر من {first.get("governorate", "")} '
            f'إلى {route["via"]} بسعر {first["price"]} جنيه، '
            f'ثم من {route["via"]} إلى {second["destination"]} '
            f'بسعر {second["price"]} جنيه. '
            f'إجمالي السعر: {route["total_price"]} جنيه'
        )
        flat_meta = {
            "via":            route["via"],
            "total_price":    route["total_price"],
            "first_station":  first.get("station", ""),
            "first_price":    first["price"],
            "second_station": second.get("station", ""),
            "second_price":   second["price"],
            "destination":    second["destination"],
        }
        docs.append(Document(page_content=content, metadata=flat_meta))
 
    return docs

##ends here

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
    route_graph: dict, # edited
    k: int = 5,
) -> list:
    """
    Three-layer retrieval strategy:

    Layer 1 — Exact metadata match:
        Extract origin + destination with the LLM, then query Pinecone
        using a metadata filter on {city, destination}. If any direct
        routes are found, return ONLY those — semantic search is skipped.
    
    Layer 2 --indirect route inference using the route graph:
        If no direct routes are found in Layer 1, use the deterministic route graph to find possible routes with one transfer. 
        This is a purely symbolic step that doesn't rely on vector search or LLM reasoning, ensuring we never hallucinate non-existent routes.
    
    Layer 3 — Semantic similarity fallback:
        Used only when Layer 1 & 2 finds nothing (no entities extracted, or
        no exact match exists in the index, no indirect routes found). Returns top-k results by
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
                print(f"[Retrieval] Layer 1: found {len(matched_docs)} direct route(s). Skipping Layer 2 & 3.")
                return matched_docs

            print("[Retrieval] Layer 1: no direct route found, falling back to indirect route search.")

        except Exception as e:
            print(f"[Retrieval] Layer 1 metadata filter failed ({e}), falling back to indirect route search.")

    # Layer 2: indirect route inference
    print("[Retrieval] Layer 2: searching for indirect routes.")
    indirect_routes = find_routes(route_graph, origin, destination)
    if indirect_routes:
        print(f"[Retrieval] Layer 2: found {len(indirect_routes)} indirect route(s).")
        # Combine and return the found routes
        return routes_to_documents(indirect_routes)
 
    print("[Routing] Graph: no routes found, trying semantic search.")

    # Layer 3: semantic similarity fallback
    print(f"[Retrieval] Layer 3: running semantic search with k={k}.")
    query_vector2 = embeddings.embed_query(query)
    return vecstore.similarity_search_by_vector(query_vector2, k=k)
    #return vecstore.similarity_search_by_vector(query, k=k)


# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    df=pd.read_excel(r'G:\chatbot\data\Sikka_data.xlsx') # load the data here to build the route graph -edited-
    route_graph=build_route_graph(df) # build the route graph once at startup -edited-

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
- إذا لم يوجد خط مباشر في المعلومات، وضّح ذلك صراحةً , بعدها اذكر الخطوط الغير مباشرةأو تبديلات للوصول عبر محطات وسيطة.
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
    app.state.route_graph = route_graph # build the route graph once at startup

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
    route_graph    = getattr(app.state, "route_graph", None)

    if not all([llm, vecstore, pinecone_index, embeddings, PROMPT,route_graph]):
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
        route_graph=route_graph # pass the route graph to the retriever -edited-
    )

    # assemble context and generate answer
    context = "\n".join(f"- {doc.page_content}" for doc in docs[:5])  # limit context to top 5 results for brevity
    filled_prompt = PROMPT.format(context=context, question=request.query)
    result = llm.invoke(filled_prompt)
    answer = result.content.strip()

    return QueryResponse(response=answer)
