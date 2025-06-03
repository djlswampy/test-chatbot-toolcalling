from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """두 정수 a와 b를 더합니다."""
    print(f"도구 실행: add({a}, {b})")
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """두 정수 a와 b를 곱합니다."""
    print(f"도구 실행: multiply({a}, {b})") 
    return a * b


# retriever를 tool로 설정
test_embedding_model = OpenAIEmbeddings()
test_vectorstore = FAISS.load_local(
    "./faiss_index_박철수",      # 인덱스 폴더 경로
    test_embedding_model,
    allow_dangerous_deserialization=True  # ← 신뢰 표시
)
test_retriever = test_vectorstore.as_retriever()

knowledge_base_tool = create_retriever_tool(
    test_retriever,
    name="knowledge_base_tool",
    description=(
        "박철수씨에 대한 정보를 제공하는 도구입니다"
    )
)