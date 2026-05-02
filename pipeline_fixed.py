# importing libraries
import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from warnings import filterwarnings
filterwarnings("ignore")

# loading environment variables
load_dotenv()

# loading data — use env var or relative path instead of hardcoded absolute path
data = pd.read_excel(r'G:\chatbot\data\Sikka_data.xlsx')


print(f"Loaded {len(data)} rows.")
print(data.head())


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


def create_documents(df):
    """Convert dataframe rows into LangChain Document objects."""
    documents = []
    for _, row in df.iterrows():
        content = (
            f'من محافظة {row["governate"]} مدينة {row["city"]} '
            f'عبر محطة {row["station_name"]} يمكن السفر إلى '
            f'{row["destination"]} بسعر {row["price"]} جنيه'
        )

        city, line_type = parse_destination(row["destination"])

        metadata = {
            "governorate": row["governate"],
            "city": row["city"],
            "station": row["station_name"],
            "destination": city,   
            "destination_gov": row['dest_gov'],   # clean city only, e.g. "الإسكندرية"
            "line_type": line_type,    # e.g. "مكيف", "بيجو", or "" (never None/list)
            "price": row["price"],
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


docs = create_documents(data)
print(f"Created {len(docs)} documents.")

# creating embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# initialising Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "chatbot-index"
EMBEDDING_DIMENSION = 384  # dimension for paraphrase-multilingual-MiniLM-L12-v2

# create index only if it doesn't exist yet
existing_indexes = [i["name"] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index '{index_name}'.")
else:
    print(f"Pinecone index '{index_name}' already exists, skipping creation.")

# upload documents to Pinecone
print("Uploading documents to Pinecone...")
PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name,
)
print("Upload complete. Pipeline finished successfully.")
