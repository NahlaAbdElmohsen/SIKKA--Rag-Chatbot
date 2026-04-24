#importing libraries
from langchain_core.documents import Document
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_pinecone
from pinecone import Pinecone, ServerlessSpec
from warnings import filterwarnings
filterwarnings("ignore")

# loading environment variables
load_dotenv()


# loading data
data=pd.read_excel('G:/chatbot/data/chatbot_data.xlsx')
data.head()

# checking data
data.info()

#cleaning data
object_cols=data.select_dtypes('object').columns
data[object_cols]=data[object_cols].apply(lambda x:x.str.strip())
data.tail()

# chunking and metadata
# def chunking(df):
#   chunks=[]
#   metadata=[]
#   for _,row in df.iterrows():
#     chunk=f"من محافظة {row['Govrnate']} مدينة {row['City']} عبر محطة {row['StationName']} يمكن السفر إلى {row['Destination']} بسعر {row['Price']} جنيه"
#     chunks.append(chunk)
#     metadata.append({'governorate':row['Govrnate'],'city':row['City'],'station':row['StationName'],'destination':row['Destination'],'price':row['Price']})
#   return chunks,metadata

# chunks,metadata=chunking(data)

def create_documents(df):
    documents = []
    for _, row in df.iterrows():
        # نص chunk: استخدام f-string مع علامات اقتباس مزدوجة
        content = f'من محافظة {row["Govrnate"]} مدينة {row["City"]} عبر محطة {row["StationName"]} يمكن السفر إلى {row["Destination"]} بسعر {row["Price"]} جنيه'
        metadata = {
            'governorate': row['Govrnate'],
            'city': row['City'],
            'station': row['StationName'],
            'destination': row['Destination'],
            'price': row['Price']
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

docs = create_documents(data)


# creating embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# creating pinecone index
pinecone_api_key=os.getenv('PINECONE_API_KEY')
pc=Pinecone(api_key=pinecone_api_key)

index_name='chatbot'
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # for OpenAI text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
#index = pc.Index("quickstart")
index = pc.Index(index_name)
