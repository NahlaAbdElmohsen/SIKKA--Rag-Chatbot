#importing libraries
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_pinecone
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from warnings import filterwarnings
filterwarnings("ignore")
from fastapi import FastAPI
from pydantic import BaseModel


load_dotenv()

# create app and models
class QueryRequest(BaseModel):
    query: str
class QueryResponse(BaseModel):
    response: str

app=FastAPI(title="Travel Chatbot API",description="API for querying the travel chatbot")



#starting up the API and loading resources

@app.on_event('startup')
async def startup_event():
  print("Starting up the travel chatbot API...")

  # set pinecone
  pinecone_api_key=os.getenv('PINECONE_API_KEY')
  pc=Pinecone(api_key=pinecone_api_key)

  # creating embeddings
  index_name='chatbot'
  embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

  # uploading data to pinecone
  vecstore=PineconeVectorStore(index_name=index_name,embedding=embeddings)
  vectorstore = LC_pinecone.from_documents(
    docs,
    embeddings,
    index_name=index_name
)

  # get gen ai response
  gemini_api_key=os.getenv('GEMINI_API_KEY')
  llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.3,api_key=gemini_api_key)

  # creating retriever and prompt template
  retriever=vecstore.as_retriever(search_kwargs={'k':3})
  prompt_template='''أنت رفيق المساعد الذكى فى السفر . استخدم المعلومات التالية للرد على استفسارات المستخدم المتعلقة بالسفر من خلال الحافلات. إذا لم تكن المعلومات كافية للإجابة، فقل "عذرًا، لا أستطيع مساعدتك في هذا الاستفسار".
     المعلومات: {context} 
     السؤال: {question} 
     الرد:'''
  PROMPT=PromptTemplate.from_template(prompt_template)

  # creating retrieval qa chain
  qa_chain=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,return_source_documents=False,chain_type_kwargs={'prompt':PROMPT})
  app.state.qa_chain=qa_chain
  print("Travel chatbot API is ready to receive queries.")


@app.post("/query",response_model=QueryResponse)
async def ask(request: QueryRequest):
  if app.state.qa_chain is None:
      return QueryResponse(response="النظام لم يكتمل تحميله بعد، حاول مرة أخرى.")
  qa_chain=app.state.qa_chain
  response=qa_chain.invoke({"query": request.query})
  answer=response['result'].strip()
  
  return QueryResponse(response=answer)


