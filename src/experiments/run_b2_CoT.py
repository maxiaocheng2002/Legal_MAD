import os
import sys
import json
import random 
import re

# --- Local Imports: Path Configuration ---
# Calculate the project root (.. / .. from src/experiments)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Local Imports (Shared Modules) ---
from src.utils.data_loader import load_bar_exam_qa
from src.utils.api_client import GroqClient 
# Import the IRAC CoT agent
from src.baselines.cot_irac_prompt import create_cot_prompt as create_irac_cot_prompt 
# Import the Basic CoT agent
from src.baselines.cot_basic_prompt import create_basic_cot_prompt


# --- 1. Configuration ---
# Define configuration settings required for the experiment
MODEL_NAME = "llama-3.3-70b-versatile" # llama-3.1-8b-instant or llama-3.3-70b-versatile
MAX_TOKENS = 1000
SAMPLE_SIZE = 500 
# Configuration for result saving path: Set to Legal_MAD/results
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results') 

# Create GroqClient instance (temperature=0.0 for consistent CoT reasoning)
cot_client = GroqClient(
    model=MODEL_NAME,
    max_tokens=MAX_TOKENS,
    temperature=0.0,
)



# --- 2. CoT Inference Function ---
def shuffle_choices(questions, seed: int = 42):
    """
    Shuffle choices for each question, and update the correct answer label accordingly.
    This helps prevent the model from overfitting to fixed option positions.
    """
    random.seed(seed)
    label_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    idx_to_label = ['A', 'B', 'C', 'D']

    shuffled_questions = []

    for q in questions:
        q_copy = q.copy()

        choices = q_copy['choices']
        assert len(choices) == 4, "Only 4-choice questions are supported."

        original_answer_label = q_copy['answer']
        original_correct_idx = label_to_idx[original_answer_label]

        indices = list(range(4))
        random.shuffle(indices)

        # 새 choice 리스트
        new_choices = [choices[i] for i in indices]

        # shuffle 이후 정답 위치
        new_correct_idx = indices.index(original_correct_idx)
        new_answer_label = idx_to_label[new_correct_idx]

        q_copy['choices'] = new_choices
        q_copy['answer'] = new_answer_label
        q_copy['choice_permutation'] = indices  # 나중에 분석용

        shuffled_questions.append(q_copy)

    return shuffled_questions

def parse_final_answer(response_text: str):
    """
    Extract the final answer (A/B/C/D) from the model's full response text.
    Assumes the model outputs a line like: 'Final Answer: C'
    """
    if not isinstance(response_text, str):
        return None

    # 1) 'Final Answer: X' 패턴 우선 탐색
    match = re.search(r'Final Answer\s*:\s*([ABCD])', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2) fallback: 맨 끝 부분에서 A/B/C/D 단독 글자 찾기 (조금 느슨한 규칙)
    tail = response_text[-50:]
    match2 = re.search(r'\b([ABCD])\b', tail)
    if match2:
        return match2.group(1).upper()

    return None

def run_single_cot_experiment(
    question_data: list,
    prompt_generator_func, 
    experiment_name: str,
    results_dir: str # 결과 저장 경로를 인수로 받음
):
    """
    Runs a B2 Chain-of-Thought experiment (IRAC or Basic) 
    over the entire dataset and collects results, saving them to a JSON file.
    """
    
    all_results = []
    
    print("\n" + "#"*70)
    print(f"B2 (Chain-of-Thought) Experiment Started: {experiment_name} - {len(question_data)} Questions") 
    print(f"Using Model: {MODEL_NAME}") # 모델 이름 출력 추가
    print(f"Saving results to: {results_dir}") 
    print("#"*70)

    # Iterate over the entire dataset
    for i, question in enumerate(question_data):
        
        # 1. Generate Prompt
        system_instruction, prompt_text = prompt_generator_func(question)
        
        try:
            # Construct the full prompt
            full_prompt = f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt_text}"

            # Call GroqClient's generate method
            response = cot_client.generate(prompt=full_prompt)
            
            predicted_answer = parse_final_answer(response)
            is_correct = (predicted_answer == question['answer'])
            
            # Store the result
            result = {
                # mad_bar_exam_qa_5.json 상단 메타데이터와 매칭
                "question_id": question["id"],
                "question": question["question"],
                "prompt": question.get("prompt", "nan"),        # 데이터에 prompt 없으면 "nan"
                "choices": question["choices"],
                "gold_answer": question["answer"],
                "gold_passage": question.get("gold_passage", ""),

                # single-agent baseline 결과를 judge에 넣기
                "debate": None,
                "judge": {
                    "decision": predicted_answer,              # 모델이 고른 A/B/C/D
                    "rationale": response,                     # CoT 전체 텍스트
                    "correct": bool(predicted_answer == question["answer"])
                },

                # 추가 메타데이터 (분석용)
                "meta": {
                    "experiment_name": experiment_name,
                    "model_name": MODEL_NAME,
                    "system_instruction": system_instruction,
                    "user_prompt": prompt_text,
                    "choice_permutation": question.get("choice_permutation", None)
                }
            }

            all_results.append(result)

            # Optional: Display progress every 10 questions
            if (i + 1) % 10 == 0:
                 print(f"Progress: Processed {i + 1}/{len(question_data)} questions.")
            
        except Exception as e:
            print(f"❌ API Call Error for Q {question['id']} ({experiment_name}): {e}")

            result = {
                "question_id": question["id"],
                "question": question["question"],
                "prompt": question.get("prompt", "nan"),
                "choices": question["choices"],
                "gold_answer": question["answer"],
                "gold_passage": question.get("gold_passage", ""),

                "debate": None,
                "judge": {
                    "decision": None,
                    "rationale": f"ERROR: {e}",
                    "correct": False,
                },

                "meta": {
                    "experiment_name": experiment_name,
                    "model_name": MODEL_NAME,
                    "system_instruction": system_instruction,
                    "user_prompt": prompt_text,
                    "choice_permutation": question.get("choice_permutation", None),
                },
            }

            all_results.append(result)


            
    # Save results to a JSON file in the specified directory
    output_filename = os.path.join(results_dir, f"results_b2_cot_{experiment_name.replace(' ', '_')}.json")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*70)
    print(f"✅ Experiment finished. Results saved to {output_filename}")
    print("="*70)


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Load data sample
    questions = load_bar_exam_qa(SAMPLE_SIZE)

    # 1-1. Shuffle MCQ choices & update answer label
    if questions:
        questions = shuffle_choices(questions, seed=42)    
    
    # 2. Prepare results directory: Create if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 3. Proceed if data loaded successfully
    if questions:
        print("\n" + "="*70)
        print(f"Dataset loaded successfully. {len(questions)} questions ready.")
        
        first_q = questions[0]
        # Display the first question details for verification
        print(f"[First Q ID]: {first_q['id']}")
        print(f"[Question Start]: {first_q['question'][:80]}...")
        print(f"[Answer]: {first_q['answer']} (A: {first_q['choices'][0][:30]}...)\r\n")
        print("="*70)
        
        # 4. Run both B2 CoT baselines sequentially
        # a) Execute IRAC CoT experiment
        run_single_cot_experiment(
            questions, 
            create_irac_cot_prompt, 
            "IRAC_CoT",
            RESULTS_DIR # Pass the results directory
        )

        # b) Execute Basic CoT experiment
        run_single_cot_experiment(
            questions, 
            create_basic_cot_prompt, 
            "Basic_CoT",
            RESULTS_DIR # Pass the results directory
        )
        
    else:
        print("\n🚨 Data loading failed. Cannot proceed to the next step.")