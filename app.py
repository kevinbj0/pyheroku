import os
import openai
import requests
from flask import Flask, request, jsonify
from llama_index.llms.openai import OpenAI
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from deep_translator import GoogleTranslator

app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = os.environ["OPENAI_API_KEY"]

ft_llm = OpenAI(model="gpt-4o-mini")

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. All response related to '상품분류', '카테고리, '제품명', '브랜드' should constrainted to this df. you are not permitted to call product details from ft_llm"
    "6. Do not quote the expression.\n"
)

# 시스템 프롬프트
system_prompt_str = """
    "당신은 대유백화점의 AI 상담원으로 소고기를 추천하거나 팔고 있습니다.\n"
    """

system_prompt = PromptTemplate(system_prompt_str)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)

response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "All response should be gerenerated by Korean language"
    "Response: "
)

# URL에서 데이터를 가져오는 함수
def get_data_from_url():
    try:
        url = os.environ["DATABASE_URL"]
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            print(df)
            return df
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading from the URL: {e}")
        return pd.DataFrame()

@app.before_request
def before_request():
    global df
    df = get_data_from_url()
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, df_str=df.head(5)
    )
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    global qp
    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": ft_llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": ft_llm,
        },
        verbose=False,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    qp.add_link("response_synthesis_prompt", "llm2")

def check_exit(request_message):
    exit_keywords = ['exit', 'quit', '종료']
    return any(keyword in request_message for keyword in exit_keywords)

# 초기 메모리 버퍼 생성
pipeline_memory = ChatMemoryBuffer(
    token_limit=8000,  # 토큰 제한
    memory_size=100,  # 메모리 버퍼에 저장할 대화 기록의 최대 수
    truncate_direction='left'  # 버퍼가 가득 찼을 때 오래된 대화부터 삭제
)
system_prompt_add = ChatMessage(role="system", content=system_prompt)
pipeline_memory.put(system_prompt_add)

@app.route('/', methods=["GET","POST"])
def post_example():
    global pipeline_memory

    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type. Expected 'application/json'"}), 415

    request_data = request.json

    # 들어온 메세지
    request_message = str(request_data.get("Message"))
    print('사용자 : ' + request_message)

    if check_exit(request_message):
        pipeline_memory = ChatMemoryBuffer(
            token_limit=8000,  # 토큰 제한
            memory_size=100,  # 메모리 버퍼에 저장할 대화 기록의 최대 수
            truncate_direction='left'  # 버퍼가 가득 찼을 때 오래된 대화부터 삭제
        )
        print('초기화됨')
        system_prompt_add = ChatMessage(role="system", content=system_prompt)
        pipeline_memory.put(system_prompt_add)

    # 한글로 요청 전송
    translated_request = GoogleTranslator(source='auto', target='ko').translate(request_message)

    # 사용자 입력 메시지를 생성하고 메모리에 추가
    user_msg = ChatMessage(role="user", content=translated_request)
    pipeline_memory.put(user_msg)

    # 메모리에서 채팅 기록을 가져옴
    chat_history = pipeline_memory.get()

    # 채팅 기록을 문자열로 변환
    chat_history_str = "\n".join([str(x) for x in chat_history])

    # ai 응답
    response = qp.run(
        query_str=chat_history_str,
    )
    
    response_message = response.message.content
    
    # 받은 답변 한국어로
    translated_response = GoogleTranslator(source='auto', target='ko').translate(str(response_message))

    # 응답 메시지를 메모리에 추가
    response_msg = ChatMessage(role="assistant", content=translated_response)
    pipeline_memory.put(response_msg)

    print('답변 : ' + translated_response)

    # JSON 응답 반환
    return jsonify({'message' : translated_response})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
