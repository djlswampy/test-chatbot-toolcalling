from dotenv import load_dotenv
# OpenAI GPT 기반 LLM 객체를 생성하는 LangChain 래퍼
from langchain_openai import ChatOpenAI
# 시스템/사용자 역할을 나눈 대화형 프롬프트 템플릿을 정의할 수 있는 클래스
from langchain_core.prompts import ChatPromptTemplate
# LLM 출력 결과를 문자열로 파싱해주는 기본 OutputParser
# AIMessage 객체를 입력으로 받아서 그 안의 content 속성만을 추출하여 문자열로 변환
from langchain_core.output_parsers import StrOutputParser
from .tools import *
from langchain_core.messages import HumanMessage
load_dotenv()

# ---- 랭체인 컴포넌트 초기화 ----
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 사용자의 질문에 적절하게 대답하는 AI 어시스턴트입니다.\n"
     "내부 문서 정보가 필요한 경우에는 'knowledgebase' 도구를 사용하여 문서를 검색하세요.\n"
     "계산이 필요한 경우 'add' 또는 'multiply' 도구를 사용하세요.\n"
     "도구 사용 없이 답할 수 있으면 직접 대답하세요.\n"
     "모든 응답은 한국어로 하세요."
    ),
    ("user", "{question}")
])

parser = StrOutputParser()

# ---- tool binding ----
tools = [add, multiply, knowledge_base_tool]
model_with_tools = llm.bind_tools(tools)


def ask(question: str) -> str:
    ai_msg = model_with_tools.invoke(question)
    print(f"tool_calls: {ai_msg.tool_calls}")
    print(f"question: {question}")

    messages = [HumanMessage(question)] # 대화 기록을 관리하기 위해 리스트 사용
    print(f"messages: {messages}")

    if not ai_msg.tool_calls:
        print("No tool calls - responding directly")
        return ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)

    # ai_msg는 llm_with_tools.invoke(messages)를 통해 받은 AIMessage 객체
    for tool_call in ai_msg.tool_calls:

        selected_tool_function = None # 선택된 도구 함수를 저장할 변수 초기화

        if tool_call["name"].lower() == "add":
            selected_tool_function = add # 'add'라는 이름의 LangChain 도구 객체
        elif tool_call["name"].lower() == "multiply":
            selected_tool_function = multiply # 'multiply'라는 이름의 LangChain 도구 객체

        # selected_tool_function에 적절한 도구 함수가 할당되었다면 (즉, 모델이 요청한 도구를 우리가 가지고 있다면)
        if selected_tool_function:
            print(f"Success: Tool {tool_call['name']}")
            # 도구 함수(selected_tool_function)에 tool_call 객체를 직접 전달하여 실행(invoke)
            # 그 결과는 자동으로 ToolMessage 객체로 포장되어 반환
            # 이 ToolMessage에는 실행 결과(content), 도구 이름(name), 그리고 원본 tool_call의 id와 동일한 tool_call_id가 포함됨
            tool_msg = selected_tool_function.invoke(tool_call)
            # 생성된 ToolMessage를 messages 리스트(대화 기록)에 추가
            # 이 messages 리스트는 나중에 모델에게 다시 전달되어 최종 답변 생성에 사용됨
            print(f"Success: tool_msg {tool_msg}")

            messages.append(ai_msg)
            messages.append(tool_msg)
        else:
            print(f"Error: Tool {tool_call['name']} not found.")

    print(messages)
    final_answer = llm.invoke(messages)
    print(f"final_answer: {final_answer}")

    return final_answer.content if hasattr(final_answer, "content") else str(final_answer)
