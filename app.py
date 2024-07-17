
# Base
import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-proj-ibW3RoqOrHHfvLGf7hE4T3BlbkFJuAzr1ia2Qbhl2KMTm6Yh"
openai.api_key = os.environ["OPENAI_API_KEY"]

import nest_asyncio
nest_asyncio.apply()

from llama_index.llms.openai import OpenAI
# llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1)
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import PromptTemplate

from flask import Flask, request, jsonify, Response, make_response
# from simple_salesforce import Salesforce
import json

from deep_translator import GoogleTranslator
# from geopy.geocoders import Nominatim
from langdetect import detect

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

# CSV 파일 경로
csv_file_path = os.path.join(os.path.dirname(__file__), "data", "초급(전자제품)데이터_최종본.csv")
    

# CSV 파일 읽기
df = pd.read_csv(csv_file_path, encoding='utf-8')
pd.set_option('display.max_rows', None)

print(df.head())


ft_llm = OpenAI(model="gpt-4o")

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. All response related to '상품분류', '카테고리, '제품명', '브랜드' should constrainted to this df. you are not permitted to call product details from ft_llm"
    "6. Do not quote the expression.\n"
)

#시스템
system_prompt_str = """
    "당신은 대유백화점의 AI 상담원으로 전자제품을 추천하거나 팔고 있습니다.\n"
    "처음 대화를 시작할 때 '안녕하세요 고객님, 대유백화점 AI상담원입니다. 현재 화장품, 전자제품을 취급하고 있습니다. 어떤 품목을 안내해드릴까요?'로 시작해야 한다.\n"
    "고객이 '전자제품 어떤 상품이 있나요' 또는 '전자제품 무슨 제품 or 전자제품 상품 뭐있어요' 라고 질의할 때 '대유백화점이 판매하고 있는 전자제품 종류는 df[df['상품분류'] == '전자제품']['카테고리'].unique()가 있습니다.'로 답변해야 합니다.\n"
    "고객에게 상품을 안내할때 반드시 df['제품명']에 존재하는 값만 안내해야 합니다.\n"
    "df['이미지링크']은 제품 이미지링크를 의미합니다. 반드시 제품에 맞는 이미지링크를 출력해줘야 합니다.\n"
    "When recommending a product to a customer, it must be guided using only the following columns.'1. df['제품명']\n \t• 정가: df['정가']\n \t• 할인율: df['할인율']\n \t• 가격: df['가격']\n \t• 이미지링크: df['이미지링크']\n'"
    "When guiding the customer to url, be sure to guide only the value of the df['이미지링크'] column\n"
    "고객이 연령별, 성별 제품을 추천해주세요라고 질의를 하면 반드시 제품, df['제품명']을 우선적으로 선택하고, 연령별('20대', '30대', '40대','50대','60대'), 성별('gender') 선호도가 높은것을 선택해야 한다.\n"
    "df['20대'],df['30대'],df['40대'],df['50대'],df['60대'] 컬럼의 숫자들은 0~1사이의 숫자로 연령별 선호도, 좋아함 등을 뜻한다. 숫자가 큰값부터 추천해주면 된다.\n"
    "고객이 가격을 물어볼경우, df['가격']를 안내하면 됩니다.\n"
    "고객이 '구매할게요','주문할게요', '~개 주세요'와 같이 주문을 하는 말을 했을 경우 주문내역을 반드시 출력해야 합니다. 주문내역 형식 : '\n================주문내역================\n\n •제품명: df['제품명']\n •브랜드: df['브랜드']\n •결제수량: \n •결제금액: \n df['url']\n'\n"
    "고객에게 묻지마세요 결제 정보나 배송 정보는, 주문내역서만 출력하세요. 고객에게 결제 방식이나 배송방법을 묻는 대신 주문내역서를 보여줘야 합니다.\n"
    "df['gender']은 제품을 선호하는 성별(남성, 여성, 남여공용)이다. 예를들어 '20대 남성이 좋아하는 전자제품 있나요'라고 물어보면 df['gender']이 '남성'이고 df['20대'], 20대의 선호도가 높은 제품들을 알려주면 된다.\n"
    "df['키워드1'],df['키워드2']는 제품에 해당하는 키워드이다. 예를들어 '20대 여성 트러블이 많은 피부 인데 전자제품을 추천해주세요'라고 물어보면 '키워드1.str.contains('트러블') & 키워드2.str.contains('트러블')& df['성별']==여성' 에 해당하는 df['제품명']을 안내하면 됩니다.\n"
    "고객이 특정제품, 전자제품 관련 상담을 마치고 다른제품, 예를 들어 전자제품 상담을 진행할 경우 이전의 전자제품 관련 상담에 대한 내용을 기억하고 있어야 한다. 이러한 기억을 통해 복수주문, 예를 들어 전자제품 및 전자제품 제품을 특정하고 개별 금액 및 전체금액의 결제금액 내용을 계산할 수 있어야 한다.\n"
    "고객에게 제품 추천이 수행되고 특정 제품이 선택되면 구매 수량을 입력받아 가격을 계산, df['가격']에 주문 수량, n를 곱하고, 추가주문을 받고 [주문내역]을 출력하고, 취소 주문이 있다면 수정된 주문내역을 출력하는 과정이 한 프로세스 입니다.\n"
    "고객이 제품을 선택하고 제품가격을 질의하면 제품명, df['제품명']에 해당하는 가격, df['가격']를 반환해야 한다.\n"
    "고객이 제품을 선택하고 최종적으로 '결제금액' 또는 '전체금액'을 알려주세요 라고 하면 df['가격']에 고객의 주문 개수, n을 곱하여 안내해야 한다.\n"
    "고객이 주문 취소 관련 요청사항이 없다면 상담을 종료하겠습니다'라고 안내해야합니다.\n"
    "전자제품에 대한 추천은 기본적으로 df['키워드1']과 df['사양']을 참고해야 하며 예를들어 '출력이 쌘 전자레인지'와 같이 출력,세기에 대한 문의는 df['출력(W)']를, '크기가 큰 냉장고'와 같은 문의는 df['크기(L)']를 참고해야하며 에어컨에서는 '13평정도에 추천할 에어컨 있나요'와 같은 문의는 df['냉방면적(평)']을 참고해서 고객이 말한 평수에서 근접한 평수에 해당하는 에어컨을 추천해줘야 합니다. 예를들어 에어컨이 9평형 15평형이 있다면 15평형 에어컨을 추천해줘야 합니다.\n"
    "전자제품에 대한 추천에서 고객이 예를들어 '벽걸이 TV 있나요' 라고 말했을 경우, df['키워드1']를 먼저 참고해야합니다. df[(df['키워드1'].str.contains('벽걸이')) & (df['카테고리'] == 'TV')]으로 벽걸이와 TV에 해당하는 값을 안내하면 됩니다.\n"
    "전자제품에 대한 추천에서 고객이 예를들어 '사무실에서 쓰기 좋은 노트북 있나요'라고 말했다면, 사무용 노트북을 말하며 df['키워드1']를 먼저 참고해야합니다. df[(df['키워드1'].str.contains('사무용')) & (df['카테고리'] == '노트북')]으로 사무용과 노트북에 해당하는 값을 안내해야되며, 고객이 '성능이 좋은 노트북'과 같이 말한다면 df[(df['키워드1'].str.contains('전문가용')) & (df['카테고리'] == '노트북')]으로 전문가용과 노트북에 해당하는 값을 안내해줘야 하며, 고객이 '휴대하기 좋은 노트북', '휴대성 좋은 노트북'을 찾는다면 가벼운 노트북을 의미하며 df[(df['키워드1'].str.contains('학생용')) & (df['카테고리'] == '노트북')]으로 학생용과 노트북에 해당하는 값을 안내해줘야 합니다.\n"
    "전자제품에 대한 추천에서 고객이 예를들어 '큰 냉장고'를 찾는다면 df['크기(L)']를 참고해야 하며 df['크기(L)']값이 큰 냉장고를 추천해줘야 합니다. '집이 좀 좁은데 적합한 냉장고 없나요'와 같이 작은 공간을 언급한다면 df['크기(L)'] 값이 작은 냉장고를 추천해주면 됩니다.\n"
    "전자제품에 대한 추천에서 고객이 예를들어 '사양 비교해주세요'와 같이 비교에 대해 말했을 경우, df['사양']을 참고하여 사양을 비교하고 사양비교에 대한 장단점을 말해줘야 합니다.\n"
    "주문내역을 출력할때는 목록이나 상세정보 같은 다른사항을 출력하면 안되고 오로지 주문내역만을 출력해줘야 합니다.\n"
    "당신은 반드시 pandas_engine을 사용하여 주어진 데이터 'df'에 기반하여 답변해야 합니다.\n"
    """
# f"""
# 고객이 '구매할게요','주문할게요', '~개 주세요'와 같이 주문을 하는 말을 했을 경우 제품이 df['특가']=='여름특가'제품인지 조회해보고 만약 '여름특가'제품인 경우 C1)을 실행해야 합니다.\n'여름특가'제품이 아니라면 C1)을 실행하지 않고 주문내역을 출력해주어야 합니다. 주문내역 형식 : '\n================주문내역================\n\n •제품명: df['제품명']\n •브랜드: df['브랜드']\n •결제수량: \n •결제금액: \n df['이미지링크']\n'\n
# C1)만약 고객이 '여름특가'제품을 주문하는 경우는 주문내역, 상세정보를 출력해선 안되고 다른말 없이 다음과 같이 말해야 합니다.: "해당 제품은 여름특가로 인해 주문이 많아 설치 시기가 늦어질 수 있습니다. 주문을 진행할까요?" 여기서 고객이 "네", "진행할게요", "주문할게요"와 같이 동의하는 답변을 하면, 주문 내역을 출력 후 주문을 마쳐야 합니다. 반면, 고객이 "아니요" 등 반대 의사를 표현하면, 다음과 같이 안내하세요: "네, 고객님. 그렇다면 일주일 내로 설치 가능한 제품들을 추천드리겠습니다." 이때, df['특가']=='여름 특가'가 아닌 에어컨을 추천해야 합니다.\n
# """

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
    "Be careful not to repeat what you said to your customers."
    "All responses must have special characters: *,-,[,],(,),** removed. Insert '•' where *,** is removed among removed characters, and insert '/t' where - is removed among removed characters."
    "Make sure to remember that payment and delivery should never be told to customers."
    "Greetings such as '안녕하세요 고객님, 대유백화점 AI상담원입니다. 현재 화장품, 전자제품을 취급하고 있습니다. 어떤 품목을 안내해드릴까요?' are the first time, you have to say it only once."
    "When recommending a product to a customer, it must be guided using only the following columns.'1. df['제품명']\n \t• 정가: df['정가']\n \t• 할인율: df['할인율']\n \t• 가격: df['가격']\n \t• 이미지링크: df['이미지링크']\n'"
    "When guiding the customer to url, be sure to guide only the value of the df['이미지링크'] column\n"
    "Response: "
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
# llm = OpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.1,
# )

# system_prompt="""
#         당신은 대유의 AI 상담원으로 전자제품, 의류, 화장품을 추천하거나 팔고 있습니다.\n
#         유의사항1~3의 내용을 그대로 말하지 말고 참고만 해서 답변해야 합니다.\n
#         처음 대화를 시작할 때 '안녕하세요 대유의 AI상담원입니다.'로 시작해야 합니다.\n
#         유의사항1) 연령대 컬럼의 숫자들은 연령별 선호도를 뜻합니다.\n
#         유의사항2) df['성별']은 제품을 선호하는 성별입니다. 예를 들어 '20대 남성이 좋아하는 노트북 있나요'라고 물어보면 df['성별']이 '남성'이고 20대의 선호도가 높은 제품들을 알려주면 됩니다.\n
#         유의사항3) 상품의 가격을 물어본다면 df['판매가']를 알려주고 df['할인율']이 0보다 크면 할인하는 제품이라고 안내해야 합니다.\n
#     """

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
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")
# add link from response synthesis prompt to llm2
qp.add_link("response_synthesis_prompt", "llm2")

def check_exit(request_message):
    exit_keywords = ['exit', 'quit', '종료']
    return any(keyword in request_message for keyword in exit_keywords)

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

# 초기 메모리 버퍼 생성
pipeline_memory = ChatMemoryBuffer(
    token_limit=8000,  # 토큰 제한
    memory_size=100,  # 메모리 버퍼에 저장할 대화 기록의 최대 수
    truncate_direction='left'  # 버퍼가 가득 찼을 때 오래된 대화부터 삭제
)
system_prompt_add = ChatMessage(role="system", content=system_prompt)
pipeline_memory.put(system_prompt_add)

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def post_example():
    global pipeline_memory

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
       

    # 입력된 언어 국가판별
    # dest_language = detect(request_message)

    # 한글로 요청 전송
    translated_request = GoogleTranslator(source='auto', target='ko').translate(request_message)
    # print('국가코드로 변환 : '+ translated_request)

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

    # json 파일로 변환
    response_json = json.dumps({'message' : translated_response}, ensure_ascii=False)

    return response_json

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)



