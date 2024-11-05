from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.chat_models import ChatOpenAI
import os

load_dotenv()

logging.langsmith("langchain_practice")

# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader("back/data/기술면접.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")
# print(docs[10].page_content)

docs[10].__dict__

# # 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")

# # 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPEN_API_KEY"))

# # 단계 4: DB 생성(Create DB) 및 저장
# # 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# # 단계 5: 검색기(Retriever) 생성
# # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# # 단계 6: 프롬프트 생성(Create Prompt)
# # 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# # 단계 7: 언어모델(LLM) 생성
# # 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), model_name="gpt-3.5-turbo", temperature=0)

# # 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
# question = "스토리북이 뭐야?"
# response = chain.invoke(question)
# print(response)