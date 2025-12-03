import os
import sys
import json

# --- Local Imports: Path Configuration ---
# Calculate the project root (.. / .. from src/experiments)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- Local Imports (Shared Modules) ---
from src.utils.data_loader import load_bar_exam_qa
from src.utils.api_client import GroqClient 
# Import the IRAC CoT agent
from src.agents.cot_irac_prompt import create_cot_prompt as create_irac_cot_prompt 
# Import the Basic CoT agent
from src.agents.cot_basic_prompt import create_basic_cot_prompt


# --- 1. Configuration ---
# Define configuration settings required for the experiment
MODEL_NAME = "llama-3.1-8b-instant" # 모델을 70b에서 8b로 변경
MAX_TOKENS = 500
SAMPLE_SIZE = 300 
# Configuration for result saving path: Set to Legal_MAD/results
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results') 

# Create GroqClient instance (temperature=0.0 for consistent CoT reasoning)
cot_client = GroqClient(model=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=0.0)


# --- 2. CoT Inference Function ---

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
            
            # Store the result
            result = {
                'id': question['id'],
                'question': question['question'],
                'expected_answer': question['answer'],
                'model_response': response,
                'experiment_name': experiment_name
            }
            all_results.append(result)

            # Optional: Display progress every 10 questions
            if (i + 1) % 10 == 0:
                 print(f"Progress: Processed {i + 1}/{len(question_data)} questions.")
            
        except Exception as e:
            # Log the error for the question
            print(f"❌ API Call Error for Q {question['id']} ({experiment_name}): {e}")
            all_results.append({
                'id': question['id'],
                'question': question['question'],
                'expected_answer': question['answer'],
                'model_response': f"ERROR: {e}",
                'experiment_name': experiment_name
            })
            
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