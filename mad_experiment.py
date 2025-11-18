import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# --- 1. 환경 설정 ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile" # 사용할 모델 ID
MAX_TOKENS = 500
client = Groq(api_key=API_KEY)

# --- 2. 데이터 로드 함수 ---
LOCAL_DATA_DIR = "data/raw/barexam_qa"
DATA_FILE_NAME = "data/qa/train.csv"

def load_bar_exam_qa_sample(sample_size: int = 300):
    # ... (Pandas를 사용한 로컬 CSV 파일 로드 로직) ...
    file_path = os.path.join(os.getcwd(), LOCAL_DATA_DIR, DATA_FILE_NAME)
    if not os.path.exists(file_path):
        print(f"❌ 오류: 로컬 데이터 파일이 존재하지 않습니다: {file_path}")
        return []
        
    try:
        df = pd.read_csv(file_path)
        data = df.to_dict(orient='records')
        sample_data = data[:sample_size]
        print(f"✅ Bar_Exam_QA 데이터 파일에서 총 {len(df)}개 중 {len(sample_data)}개 문제 로드 완료.")
        return sample_data
    except Exception as e:
        print(f"❌ 오류: 로컬 파일 로드 중 오류 발생: {e}")
        return []

# --- 3. CoT/MAD 추론 함수 (다음 단계에서 구현 예정) ---
def run_cot_baseline(question_data):
    """B2 Chain-of-Thought 베이스라인을 실행하는 함수"""
    pass # 여기에 Groq API 호출 로직이 들어갑니다.

# --- 4. 메인 실행 블록 ---
if __name__ == "__main__":
    questions = load_bar_exam_qa_sample(300)
    if questions:
        print("\n" + "="*70)
        print(f"데이터셋 로드 성공. 총 {len(questions)}개 문제 준비 완료.")
        run_cot_baseline(questions)
    else:
        print("\n🚨 데이터 로드 실패.")