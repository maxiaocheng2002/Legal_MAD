import os
import json
import re

# 프로젝트 루트 경로 설정 (run_c1_evaluation.py가 src/experiments 안에 있으므로)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
EVALUATION_FILENAME = "evaluation_results_b2.json"

# --- 1. 정답 추출 함수 ---
def extract_answer(model_response: str) -> str or None:
    """
    모델 응답 텍스트에서 최종 정답 (A, B, C, D)를 추출합니다.
    """
    # 1. "Final Answer: (X)" 형태를 찾습니다.
    match_final = re.search(r'Final\s+Answer:\s*\(?([A-D])\)?', model_response, re.IGNORECASE)
    if match_final:
        return match_final.group(1).upper()

    # 2. 응답 끝에서 (X) 형태를 찾습니다.
    match_end = re.search(r'\([A-D]\)\s*$', model_response.strip())
    if match_end:
        return match_end.group(0).strip('()').upper()
    
    # 3. 만약 응답이 단순한 A, B, C, D 문자 하나만 포함한다면 그것을 반환
    if model_response.strip().upper() in ['A', 'B', 'C', 'D']:
         return model_response.strip().upper()

    return None

# --- 2. 평가 실행 함수 ---
def run_evaluation(experiment_name: str) -> dict:
    """
    단일 실험 결과 파일에 대한 정확도를 계산합니다.
    """
    filename = os.path.join(RESULTS_DIR, f"results_b2_cot_{experiment_name.replace(' ', '_')}.json")
    
    if not os.path.exists(filename):
        print(f"🚨 Error: Result file not found: {filename}")
        return {"total": 0, "correct": 0, "accuracy": 0.0, "error": "File Not Found"}

    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    total_questions = len(results)
    correct_answers = 0
    
    for item in results:
        expected = item.get('expected_answer', 'X').strip().upper()
        
        # 모델 응답이 텍스트가 아닌 경우 (예: API 에러) 건너뜁니다.
        model_response = item.get('model_response')
        if not isinstance(model_response, str):
            continue 
            
        predicted = extract_answer(model_response)
        
        # 정답이 유효하고 예측된 정답이 정답과 일치하는 경우
        if expected in ['A', 'B', 'C', 'D'] and predicted == expected:
            correct_answers += 1
            
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0.0
    
    return {
        "experiment_name": experiment_name,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": f"{accuracy:.2f}%"
    }

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    
    experiments = ["IRAC_CoT", "Basic_CoT"]
    all_evaluation_results = {}
    
    print("\n" + "="*70)
    print("🚀 C1 - Evaluation Process Started")
    print(f"Loading results from: {RESULTS_DIR}")
    print("="*70)
    
    for exp in experiments:
        print(f"  -> Evaluating {exp}...")
        results = run_evaluation(exp)
        all_evaluation_results[exp] = results
        
        print(f"     ✅ {exp} Accuracy: {results.get('accuracy', 'N/A')}")
        print(f"     [Correct: {results.get('correct_answers')}/{results.get('total_questions')}]")
        print("-" * 50)
        
    # 최종 결과를 별도의 JSON 파일로 저장
    output_filename = os.path.join(RESULTS_DIR, EVALUATION_FILENAME)
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_evaluation_results, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*70)
    print("🎉 Evaluation Complete!")
    print(f"Detailed evaluation results saved to: {output_filename}")
    print("="*70)

