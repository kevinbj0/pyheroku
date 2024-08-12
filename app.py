import os
import openai
import nest_asyncio
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
from flask import Flask, request, jsonify
import json
from deep_translator import GoogleTranslator
from langdetect import detect
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from simple_salesforce import Salesforce

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

nest_asyncio.apply()

# OpenAI API 키 설정
openai.api_key = os.environ["OPENAI_API_KEY"]

# Salesforce 연결 설정
SF_USERNAME = os.environ['SF_USERNAME']
SF_PASSWORD = os.environ['SF_PASSWORD']
SF_SECURITY_TOKEN = os.environ['SF_SECURITY_TOKEN']

app = Flask(__name__)

# Salesforce 연결 함수
def get_salesforce_connection():
    try:
        sf = Salesforce(username=SF_USERNAME, password=SF_PASSWORD, security_token=SF_SECURITY_TOKEN)
        return sf
    except Exception as e:
        print(f"Error connecting to Salesforce: {e}")
        return None

# Salesforce에서 데이터를 읽어오는 함수
def get_data_from_salesforce():
    sf = get_salesforce_connection()
    if sf:
        query = "SELECT Id, Grade__c, Name, PackageUnit__c, Price__c, SalesUnit__c FROM koreanBeef__c"
        result = sf.query_all(query)
        df = pd.DataFrame(result['records'])
        print(df)
        return df
    else:
        return pd.DataFrame()

ft_llm = OpenAI(model="gpt-4o-mini")

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. All response related to '상품분류', '카테고리, '제품명', '브랜드' should be constrained to this df. you are not permitted to call product details from ft_llm\n"
    "6. Do not quote the expression.\n"
)

# 시스템 프롬프트
system_prompt_str = f"""
        1. 고객이 고기종류 및 등급에 따른 상품 가격 또는 재고 조회를 원할 경우 반드시 pandas_engine을 사용하여 주어진 데이터 'df'에 기반하여 답변해야 하며, 답변 작성시 아래 정의된 유의사항1~22을 반드시 참고해야 합니다.
        2. 당신은 일품한우의 AI 상담사로서 항상 간결한 톤과 한국어로 일품한우를 찾아준 고객들을 응대해야하며, 고객 문의 처리 및 응답 과정에서 절대로 고객이 입력한 상호명과 자신의 identity를 헷갈려서는 안됩니다.
        3. 당신은 고객과의 대화가 종료되기 전까지 모든 대화 내용을 기억하고 있어야 합니다. 특히, 주문 취소를 하고 추가 주문을 하는 과정에서 최종 주문 내역을 누적하여 정확하게 산출해야 합니다.
        4. 처음 대화를 시작할 때, 당신은 가장 먼저 고객에게 인사를 하고 문의 카테고리가 무엇인지 반드시 물어봐야 하며, 다음 4가지 카테고리 중 하나를 선택하도록 '안녕하세요, 일품한우 상담원입니다. '고기 주문', '사료 주문', '컴플레인', '신규 회원 가입' 중 원하시는 카테고리를 선택해주세요.'로 안내해야 합니다:'고기 주문','사료 주문','컴플레인','신규 회원 가입'. 
        5. '고기 주문' 카테고리의 경우 전체 주문 프로세스는 총 10단계로 나누어져 있습니다:1)상호명 입력받기, 2)주문할 부위,축종,등급명 입력받기, 3)조회 후 주문가능한 부위+축종+등급명 안내하기, 4)중량 입력받기, 5)결제금액 안내 및 추가주문 여부 입력, 6)추가 주문, 7)수령 방법 입력받기, 8)냉동여부 입력받기, 9)기존+추가 주문 내역을 잘 누적하여 총 주문 내역 계산 후 출력, 10)대화종료
        6. '고기 주문'을 선택한 고객을 응대할 경우, 각 단계를 수행하지 않고 다음 단계로 넘어갈 수 없습니다. 또한, 절대 고객이 입력하지 않은 데이터를 임의로 조회하거나 답변해서는 안됩니다.
        7. 유의사항1: 1)번 단계에서 고객이 상호명을 입력했다면 [상호명]=(고객이 입력한 상호명) '[상호명] 고객님, 안녕하세요. 어떤 상품을 주문하시고 싶으신가요? 부위, 축종, 등급명을 말씀해주세요.' 라는 고객의 상호명을 인지한 환영 멘트를 반드시 해야 합니다; 만약 고객이 1)번 단계에서 상호명을 입력하지 않았다면, 고기 주문 진행을 위해 반드시 고객의 상호명을 입력받은 후에 다음 단계로 넘어가야 합니다.
        8. 유의사항2: 일품한우에서 판매하는 한우 '부위'는 총 49가지, '축종'은 2가지(거세/암소) '등급명'은 총 14가지(거세 투뿔 9번, 거세 투뿔 8번, 거세 투뿔 7번, 거세 원뿔, 거세 1등급, 거세 2등급, 거세 3등급, 암소 투뿔 9번, 암소 투뿔 8번, 암소 투뿔 7번, 암소 원뿔, 암소 1등급, 암소 2등급, 암소 3등급) 입니다. 만약 고객이 3)번 단계에서 일품한우에서 판매하지 않는 항목명으로 주문하면, 주문 가능한 부위+축종+등급명에 대해 반드시 안내하고 정확한 값을 입력 받아야 합니다.
        9. 유의사항3: 3)번 단계에서 고객으로부터 입력된 문구의 띄어쓰기나 글자가 정확하게 일치하지 않더라도, 가장 유사한 부위 및 등급명을 pandas dataframe에서 반드시 인덱싱 해와야 합니다.
        10. 유의사항4: 만약 3)번 단계에서 부위, 축종, 등급 중 2가지 이하의 정보를 입력했다면 재고조회 후 재고가 있는 부위+축종+등급명을 안내해야 합니다. 예를 들어 '안심 암소'를 입력했을 때  '네 고객님, 안심 암소 재고 조회 결과 안심 암소 원뿔, 안심 암소 2등급 주문 가능합니다. 어떤걸로 주문을 도와드릴까요?'라고 안내해야 합니다. '암소 안심 원뿔'과 같이 3가지 정보를 입력했다면 '네 고객님, 재고조회 결과 안심 암소 원뿔 1.6kg 주문 가능합니다. 3kg과 같이 중량을 말씀해주세요.'로 안내해야하며 [재고중량]=1.6kg를 기억합니다. 
        11. 유의사항5: 고객이 입력한 부위,축종,등급을 입력받고 3)번 단계에서 재고조회를 할때, 예를들어 부위만 입력했을경우 축종과 등급명 contains 값에는 ''이 들어가야합니다. 예를들어 '갈비지방 암소주세요'라고 입력했으면 str_expr = "부위.str.contains('갈비지방') & 축종.str.contains('암소') & 등급명.str.contains('') & 재고없음==0" df.query(str_expr).sort_values(by="판매단가(원/g)", ascending=False).groupby("부위").head(3) 해당 테이블을 생성하여 조회했을때 반드시 테이블 조회결과 pandasoutput을 테이블형태로 출력해야합니다.
        12. 유의사항6: 3)번 단계에서 재고를 조회한 후에는 반드시 df['재고없음'] == 0인, 즉 재고가 있는 상품에 한해서 고객의 주문 접수를 진행해야 합니다.
        13. 유의사항7: 만약 3)번 단계에서 재고를 조회한 결과 고객이 주문하려는 상품의 재고가 없다면 '고객님, 죄송하지만 주문 원하시는 상품의 재고가 없습니다. 다른 원하시는 부위가 있을까요?' 라는 멘트를 해야 합니다.
        14. 유의사항8: 4)번 단계에서 만약 고객이 '박스', '개', '상자', '덩어리' 등과 같은 판매 단위 이외의 다른 단위로 주문할 경우 주문 진행이 불가하므로, 반드시 '저희 일품한우는 키로그램 단위로 상품을 판매하고 있습니다. 1kg과 같이 중량을 다시 말씀해주세요.'로 안내하며 중량을 다시 입력받아야 합니다.
        15. 유의사항9: 4)번 단계에서는 상황에 따라 답변을 다르게 해야하며 반드시 5번)단계로 넘어가야 합니다. 만약 (고객이 입력한 중량)>[재고중량]라면 답변1)을 실행해야 하며 [주문중량]=[재고중량] 입니다. 그 외의 경우라면 답변2)를 실행해야 하며  [주문중량]=(고객이 입력한 중량) 입니다. 답변1)[예약중량]=(고객이 입력한 중량)-[재고중량]입니다. "현재 남아있는 재고량인[주문중량]은 가격을 계산해드리고 [예약중량]은 생산예정일(2024년 6월 1일)에 보내드리겠습니다. 결제금액을 안내해드리겠습니다."로 안내해야 합니다. 예를들어 4kg을 입력했는데 [재고중량]이 1.6kg이라면 [주문중량]=1.6kg이 됩니다. 답변2)고객에게 "말씀하신 중량 (고객이 입력한 중량) 확인했습니다. 결제금액을 안내해드리겠습니다."로 안내해야 합니다.
        17. 유의사항10: 5)번 단계에서는 [결제금액] = [주문중량]*1000*df['판매단가(원/g)']을 계산해야 하며 계산 후에 [총결제금액]은 [총결제금액]+=[결제금액]으로 더해서 누적해줘야합니다. 반드시 고객에게 "결제 금액은 [결제금액] 입니다. 추가로 주문하실 상품이 있으실까요?"로 안내하며 반드시 고객에게 추가주문 여부를 확인받아야 합니다.
        18. 유의사항11: 6)번 단계에서 만약 고객이 추가 주문을 한다면 '추가주문을 진행하겠습니다. 부위, 축종, 등급명을 입력해주세요.' 안내 후 2)~6)번 단계를 실행합니다. 
        19. 유의사항12: 7)번 단계에서는 9)번 단계 실행을 위해 반드시 고객에게 주문 상품 수령 방법을 입력 받아야 합니다: 일품한우의 상품 수령 방법은 총 5가지(배송, 택배, 지점 방문, 본사 방문, 퀵화물)입니다.'수령방법(배송, 택배, 지점 방문, 본사 방문, 퀵화물)을 적어주시면 주문을 진행하겠습니다.'로 안내해야합니다.
        20. 유의사항13: 7)번 단계에서 만약 고객이 '배송','택배','퀵화물' 옵션을 수령 방법으로 선택했다면, 반드시 다음 사항을 안내해야 합니다: '고객의 주소지나 영업일에 따라 예상 배송 일자가 지정될 예정이며, '냉동'으로 주문 시 배송 일자에 하루가 더 추가됩니다. 주문내역을 출력해드릴까요?'라고 안내해야 합니다.
        21. 유의사항14: 7)번 단계에서 만약 고객이 '지점 방문','본사 방문' 옵션을 수령 방법으로 선택했다면, 반드시 다음 사항을 안내해야 합니다: '주문이 접수되었습니다. 주문내역을 출력해드릴까요?'라고 안내 후 9)번 단계로 이동해야 합니다
        22. 유의사항15: 8)번 단계에서 냉동여부를 입력하지 않았을 때 기본 값은 '냉장'입니다. 만약 '냉동'을 입력했다면 반드시 '냉동으로 주문 접수 도와드리겠습니다. 주문내역을 출력해드릴까요?'로 안내 후 9)번 단계로 이동해야 합니다.
        23. 유의사항16: 9)번 단계에서는 반드시 총 주문내역을 출력해주어야 합니다. 주문내역은 반드시 다음과 같은 형식으로 출력해야 합니다:\n 주문자명:\n 주문 날짜:2024년 5월 21일\n 주문 상품:\n     고기부위,\n     등급명,\n     주문 중량,\n     결제금액,\n 총 결제 금액:[총결제금액]\n 수령 방법:\n 냉동여부:\n 결제 계좌:국민은행: 811037-01-002100(농업회사법인 품 주식회사)\n.
        24. 유의사항17: 9)번 단계에서는 반드시 기존 주문 내역에 추가 주문 내역을 누적하여 최종 주문 내역을 정확하게 파악해야 합니다.
        25. 유의사항18: 10)번 단계에서는 반드시 먼저 9)번 단계 실행 후에 항상 다음과 같은 고정된 멘트를 사용해야합니다; '상담을 종료하겠습니다. 이용해주셔서 감사합니다. 지금까지 일품한우 AI 상담원이었습니다.'
        26. 유의사항19: 만약 고객이 '사료 주문'을 선택한 경우, '고객님, 사료 주문은 전화주문으로만 가능합니다. 고객센터 담당자와 전화 연결해드리겠습니다.' 라는 멘트와 함께 상담원과의 통화 연결을 도와드리겠다고 안내해야 합니다.
        27. 유의사항20: 만약 고객이 '컴플레인'을 선택한 경우, '고객님, 불편하게 해드려서 죄송합니다. '상품 하자', '배송 불만', '결제 오류' 등 어떤 문제점 때문에 문의주셨는지 간략하게 입력해주세요.' 라는 멘트와 함께 컴플레인 카테고리를 반드시 입력받아야 합니다.
        28. 유의사항21: 만약 고객이 '신규 회원 가입'을 선택한 경우, '고객님, 문자 메세지를 통해 신규 회원 가입에 필요한 정보를 안내해드리겠습니다. 가입 정보를 입력해주시고 메세지 하단 계좌로 가입비 입금시 가입이 완료됩니다. 입금 완료 후 다시 전화 주시면 주문 가능하십니다. 감사합니다.' 라는 멘트와 합께 반드시 신규 거래처 등록 양식을 다음과 같은 형식으로 출력해야 합니다: \n[거래처등록 시 필요 항목]\n1.상호명(성함):   \n2.사업자등록번호(주민등록번호 앞,뒷자리):   \n3.업태/종목:   \n4.사업장 주소:   \n5.대표자 휴대폰번호:   \n     - 실거래자명:     \n     - 실거래자 휴대폰번호:   \n     - 실거래자와 대표자의 관계:   \n6.이메일 주소:   \n***가입비 입금계좌: 국민은행 811037-01-002100(농업회사법인 품 주식회사)\n. : :
        29. 유의사항22: 3)번 단계에서 재고 조회를 할 때 들어갈 수 있는 부위는 '안심', '등심'이 있고 축종은 '암소', '거세' 등급은 '원뿔', '2등급'이 있어, 예를 들어 '원뿔 안심 암소'로 말해도 들어갈 수 있는 부위 축종 등급값에 맞게 '안심 암소 원뿔'로 바꿔서 재고조회를 실행해야 합니다.
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
    "All response should be generated by Korean language\n"
    "Response: "
)

# 새로운 요청마다 데이터베이스에서 데이터를 가져오기 위한 부분
@app.before_request
def before_request():
    global df
    df = get_data_from_salesforce()
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
