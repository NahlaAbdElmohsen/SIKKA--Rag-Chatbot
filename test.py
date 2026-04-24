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

# loading environment variables
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
            "destination_gov": row['dest_gov'], # clean city only, e.g. "الإسكندرية"  imp here how it will be used in the retrieval step to find indirect routes based on the destination governorate
            "line_type":   line_type,
            "price":       row["price"],       
            "station":     row["station_name"],
            "governorate": row["governate"],
        })
    return graph

def find_routes(graph: dict, origin: str, destination: str) -> dict:
    """
    2-hop BFS — returns routes with exactly one transfer.
    This is a purely symbolic step that doesn't rely on vector search or LLM reasoning, ensuring we never hallucinate non-existent routes.
    """
    results = []
 
    # 1-transfer legs: origin → hub → destination
    for first_leg in graph.get(origin, []):
        hub = first_leg["destination"]
        if hub == destination:
            continue            # captured as direct
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