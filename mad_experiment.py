import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# --- Import Agent Classes ---
from src.agents.judge import JudgeAgent
from src.agents.debater import DebaterAgent

# --- 1. Configuration ---
# Load environment variables from the .env file.
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Check your .env file.")

# Model specified in project requirements
MODEL_NAME = "llama-3.3-70b-versatile" 
MAX_TOKENS = 500

client = Groq(api_key=API_KEY)

# --- 2. Data Loading Function ---
# Local path to the downloaded Bar Exam QA dataset
LOCAL_DATA_DIR = "data/raw/barexam_qa"
DATA_FILE_NAME = "data/qa/train.csv"

def load_bar_exam_qa_sample(sample_size: int = 300):
    """
    Loads the Bar_Exam_QA CSV file directly using Pandas and returns a sample.
    """
    # Construct the full path to the local data file.
    file_path = os.path.join(os.getcwd(), LOCAL_DATA_DIR, DATA_FILE_NAME)
    
    if not os.path.exists(file_path):
        # ERROR: Data file is missing.
        print(f"❌ ERROR: Local data file does not exist: {file_path}")
        return []
        
    try:
        # Load the CSV file into a Pandas DataFrame.
        df = pd.read_csv(file_path)
        # Convert DataFrame records to a list of dictionaries.
        data = df.to_dict(orient='records')
        # Sample the data.
        sample_data = data[:sample_size]
        
        print(f"✅ Bar_Exam_QA data successfully loaded: {len(sample_data)} out of {len(df)} questions loaded.")
        return sample_data
        
    except Exception as e:
        print(f"❌ ERROR: An error occurred while loading the local file: {e}")
        return []

# --- 3. CoT/MAD Inference Functions ---

def create_cot_prompt(question):
    """Creates the Chain-of-Thought prompt text based on the IRAC framework."""
    
    question_text = f"**Question**: {question['question']}\n"
    choices = (f"(A) {question['choice_a']}\n(B) {question['choice_b']}\n"
               f"(C) {question['choice_c']}\n(D) {question['choice_d']}")
    
    # System Instruction (The core CoT/IRAC instruction)
    system_instruction = (
        "You are a legal expert with the highest logic and accuracy. Analyze the legal problem "
        "and strictly follow the IRAC (Issue, Rule, Application, Conclusion) methodology for step-by-step reasoning. "
        "The final conclusion (Final Answer) must be one of the given choices (A, B, C, D)."
    )
    
    # User prompt template enforcing the IRAC structure
    prompt = (
        f"Analyze the legal question and choices, then derive the correct answer through the IRAC reasoning process:\n\n"
        f"--- Problem ---\n{question_text}"
        f"\n--- Choices ---\n{choices}"
        f"\n----------------\n\n"
        f"ASSISTANT (Chain-of-Thought):\n"
        f"1. Issue: [Identify the core legal issue here.]\n"
        f"2. Rule: [State the applicable laws, legal principles, and precedents here.]\n"
        f"3. Application: [Apply the rule to the facts of the problem with logical analysis.]\n"
        f"4. Conclusion: [State the final conclusion here.]\n\n"
        f"Final Answer: [Write only the correct choice (A/B/C/D) here.]"
    )
    
    return system_instruction, prompt

def run_cot_baseline(question_data):
    """Runs the B2 Chain-of-Thought baseline for the first sample question."""
    
    # Use only the first question for the initial test
    test_question = question_data[0]
    
    system_instruction, prompt_text = create_cot_prompt(test_question)
    
    print("\n" + "#"*70)
    print(f"B2 (Chain-of-Thought) Baseline Test Started: {test_question['idx']}")
    print("#"*70)

    try:
        # Groq API Call (Model: llama-3.3-70b-versatile)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_text},
            ],
            model=MODEL_NAME, 
            temperature=0.0, # Set to 0.0 for consistent legal reasoning
            max_tokens=MAX_TOKENS,
        )
        
        response = chat_completion.choices[0].message.content
        
        print("\n[Groq Llama 3.3 70B CoT Response]")
        print(response)
        print("\n" + "="*70)
        
        # Display expected answer
        expected_answer = test_question['answer']
        print(f"[Expected Answer]: {expected_answer}")
        
    except Exception as e:
        print(f"❌ API Call Error: {e}")

def run_mad_baseline(question_data):
    """Runs the B3 Multi-Agent Debate (MAD) baseline for the first sample question."""
    
    test_question = question_data[0]
    
    print("\n" + "="*70)
    print(f"B3 (Multi-Agent Debate) Baseline Test Started: {test_question['idx']}")
    print("="*70)

    # 1. Debater Agents Initialization
    # Debater A: Supports choices A or C
    debater_a = DebaterAgent(model_name=MODEL_NAME, stance='A', temperature=0.7)
    # Debater B: Supports choices B or D
    debater_b = DebaterAgent(model_name=MODEL_NAME, stance='B', temperature=0.7)
    
    # 2. Judge Agent Initialization (Temperature 0.0 for impartiality)
    judge = JudgeAgent(model_name=MODEL_NAME, temperature=0.0)

    try:
        # 3. Argument Generation (The Debate)
        print("Generating Argument A (Supports A/C)...")
        # Debater A supports A and C
        argument_a = debater_a.generate_argument(test_question, ('A', 'C'))
        
        print("\nGenerating Argument B (Supports B/D)...")
        # Debater B supports B and D
        argument_b = debater_b.generate_argument(test_question, ('B', 'D'))
        
        arguments = {'A': argument_a, 'B': argument_b}

        # 4. Judge Ruling
        print("\nJudge Agent is evaluating arguments and generating the final ruling...")
        ruling = judge.generate_ruling(test_question, arguments)

        # 5. Result Display
        print("\n" + "*"*70)
        print("[MAD Debate Summary]")
        print("*"*70)
        print(f"\n[Argument A]:\n{argument_a[:300]}...\n") # Print first 300 characters of Argument A
        print(f"\n[Argument B]:\n{argument_b[:300]}...\n") # Print first 300 characters of Argument B
        print("\n" + "-"*70)
        print("[JUDGE FINAL RULING]")
        print(ruling)
        print("\n" + "="*70)
        
        # Display expected answer
        expected_answer = test_question['answer']
        print(f"[Expected Answer]: {expected_answer}")

    except Exception as e:
        print(f"❌ MAD Execution Error: {e}")

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    SAMPLE_SIZE = 300
    
    # 1. Load data
    questions = load_bar_exam_qa_sample(SAMPLE_SIZE)
    
    # 2. Proceed to the next step if data loaded successfully
    if questions:
        print("\n" + "="*70)
        print(f"Dataset loaded successfully. {len(questions)} questions ready.")
        
        first_q = questions[0]
        print(f"[First Q ID]: {first_q['idx']}")
        print(f"[Question Start]: {first_q['question'][:80]}...")
        print(f"[Answer]: {first_q['answer']} (A: {first_q['choice_a'][:30]}...)")
        print("="*70)
        
        # 3. Run the B3 MAD baseline
        # You may comment out run_cot_baseline(questions) if you only want to run MAD
        run_mad_baseline(questions) 
    else:
        print("\n🚨 Data loading failed. Cannot proceed to the next step.")