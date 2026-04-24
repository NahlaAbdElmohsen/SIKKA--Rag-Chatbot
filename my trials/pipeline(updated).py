# importing libraries
import os
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
data = pd.read_excel('G:\chatbot\data\Sikka_data.xlsx')

# cleaning data
object_cols = data.select_dtypes("object").columns
data[object_cols] = data[object_cols].apply(lambda x: x.str.strip())
#data['Price'] = data['Price'].astype(float) # ensure price is float for better handling later

print(f"Loaded {len(data)} rows.")
print(data.head())


def create_documents(df):
    """Convert dataframe rows into LangChain Document objects."""
    documents = []
    for _, row in df.iterrows():
        content = (
            f'من محافظة {row["governate"]} مدينة {row["city"]} '
            f'عبر محطة {row["station_name"]} يمكن السفر إلى '
            f'{row["destination"]} بسعر {row["price"]} جنيه'
        )
        metadata = {
            "governorate": row["governate"],
            "city": row["city"],
            "station": row["station_name"],
            "destination": row["destination"].strip().split()[0],  # take first word of destination for better matching
            'line_type':row['destination'].strip().split()[1:-1],  # e.g. "الفيوم (مكيف)" -> "مكيف"
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

# upload documents to Pinecone — this is the step that was missing
print("Uploading documents to Pinecone...")
PineconeVectorStore.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name=index_name,
)
print("Upload complete. Pipeline finished successfully.")
