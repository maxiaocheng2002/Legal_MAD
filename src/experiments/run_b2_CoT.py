import os
import sys

# --- Local Imports: Path Configuration ---
# Add the parent directory (root) to the path to access 'src/utils' and 'src/agents' modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Local Imports (Shared Modules) ---
from src.utils.data_loader import load_bar_exam_qa
from src.utils.api_client import GroqClient 
# Import the IRAC CoT agent (Renamed file)
from src.agents.cot_irac_prompt import create_cot_prompt as create_irac_cot_prompt 
# Import the Basic CoT agent (Newly created file)
from src.agents.cot_basic_prompt import create_basic_cot_prompt


# --- 1. Configuration ---
# Define configuration settings required for the experiment
MODEL_NAME = "llama-3.3-70b-versatile" 
MAX_TOKENS = 500
SAMPLE_SIZE = 300 

# Create GroqClient instance (temperature=0.0 for consistent CoT reasoning)
cot_client = GroqClient(model=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=0.0)


# --- 2. CoT Inference Function ---

def run_single_cot_experiment(
    question_data, 
    prompt_generator_func, 
    experiment_name: str
):
    """
    Runs a single B2 Chain-of-Thought experiment (IRAC or Basic) using the configured GroqClient.
    
    Args:
        question_data: List of question dictionaries.
        prompt_generator_func: The function (e.g., create_irac_cot_prompt) that generates the prompt.
        experiment_name: Descriptive name for the current experiment run.
    """
    
    # Use only the first question for the initial demonstration/test
    test_question = question_data[0]
    
    # 1. Generate Prompt (IRAC or Basic)
    system_instruction, prompt_text = prompt_generator_func(test_question)
    
    print("\n" + "#"*70)
    # FIX: Changed 'idx' to 'id' to match data_loader.py output
    print(f"B2 (Chain-of-Thought) Test Started: {experiment_name} - Q: {test_question['id']}") 
    print("#"*70)

    try:
        # Construct the full prompt including the System Instruction
        full_prompt = f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt_text}"

        # Call GroqClient's generate method
        response = cot_client.generate(prompt=full_prompt)
        
        print(f"\n[Groq Llama 3.3 70B {experiment_name} Response]")
        print(response)
        print("\n" + "="*70)
        
        # Display expected answer
        expected_answer = test_question['answer']
        print(f"[Expected Answer]: {expected_answer}")
        
    except Exception as e:
        print(f"❌ API Call Error for {experiment_name}: {e}")


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    
    # 1. Load data sample
    questions = load_bar_exam_qa(SAMPLE_SIZE)
    
    # 2. Proceed if data loaded successfully
    if questions:
        print("\n" + "="*70)
        print(f"Dataset loaded successfully. {len(questions)} questions ready.")
        
        first_q = questions[0]
        # FIX: Changed 'idx' to 'id' to match data_loader.py output
        print(f"[First Q ID]: {first_q['id']}")
        print(f"[Question Start]: {first_q['question'][:80]}...")
        # FIX: Changed 'choice_a' to 'choices[0]' to access the first item in the choices list
        print(f"[Answer]: {first_q['answer']} (A: {first_q['choices'][0][:30]}...)")
        print("="*70)
        
        # 3. Run both B2 CoT baselines sequentially

        # a) Execute IRAC CoT experiment
        run_single_cot_experiment(
            questions, 
            create_irac_cot_prompt, 
            "IRAC CoT (With IRAC Structure)"
        )

        # b) Execute Basic CoT experiment
        run_single_cot_experiment(
            questions, 
            create_basic_cot_prompt, 
            "Basic CoT (Without IRAC Structure)"
        )
        
    else:
        print("\n🚨 Data loading failed. Cannot proceed to the next step.")