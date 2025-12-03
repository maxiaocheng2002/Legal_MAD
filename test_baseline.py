"""
Quick test script for B1: Single-Agent Baseline
Run this after adding API key to .env

Usage:
    python test_baseline.py [provider]
    
    provider: "groq" (default) or "openrouter"
    
Examples:
    python test_baseline.py          # Uses Groq
    python test_baseline.py groq     # Uses Groq
    python test_baseline.py openrouter  # Uses OpenRouter
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.api_client import GroqClient
from src.baselines.single_agent import SingleAgentBaseline


def test_api_connection():
    """Test basic API connection."""
    print("=" * 60)
    print("TEST 1: API Connection (GROQ)")
    print("=" * 60)
    
    try:
        client = GroqClient()
        response = client.generate("Say 'API is working!' in one sentence.")
        print(f"[PASS] API Response: {response}")
        return True
    except Exception as e:
        print(f"[FAIL] API Connection Failed: {e}")
        print("\nMake sure you:")
        print("1. Added GROQ_API_KEY to .env file")
        print("2. Got your key from https://console.groq.com/")
        return False


def test_json_generation():
    """Test JSON response generation."""
    print("\n" + "=" * 60)
    print("TEST 2: JSON Generation (GROQ)")
    print("=" * 60)
    
    try:
        client = GroqClient()
        prompt = """
        Respond in JSON format:
        {
            "status": "success",
            "message": "JSON generation is working"
        }
        """
        response = client.generate_json(prompt)
        print(f"[PASS] JSON Response: {response}")
        return True
    except Exception as e:
        print(f"[FAIL] JSON Generation Failed: {e}")
        return False


def test_mcq_baseline():
    """Test MCQ baseline."""
    print("\n" + "=" * 60)
    print("TEST 3: MCQ Baseline (GROQ)")
    print("=" * 60)
    
    try:
        baseline = SingleAgentBaseline()
        
        # Simple test question
        question = "Which amendment to the US Constitution protects freedom of speech?"
        choices = [
            "The First Amendment",
            "The Second Amendment",
            "The Fifth Amendment",
            "The Tenth Amendment"
        ]
        
        print(f"Question: {question}")
        print(f"Choices: {choices}")
        print("\nGenerating answer...")
        
        result = baseline.answer_mcq(
            question=question,
            prompt_context="",
            choices=choices
        )
        
        print(f"\n[PASS] Answer: {result['answer']}")
        print(f"Reasoning: {result['reasoning'][:200]}...")
        print(f"Citations: {result.get('citations', [])}")
        return True
        
    except Exception as e:
        print(f"[FAIL] MCQ Baseline Failed: {e}")
        return False


def test_open_ended_baseline():
    """Test open-ended baseline."""
    print("\n" + "=" * 60)
    print("TEST 4: Open-Ended Baseline (GROQ)")
    print("=" * 60)
    
    try:
        baseline = SingleAgentBaseline()
        
        question = "What are the key elements required to establish a valid contract?"
        
        print(f"Question: {question}")
        print("\nGenerating answer...")
        
        result = baseline.answer_open_ended(question=question)
        
        print(f"\n[PASS] Answer: {result['answer'][:200]}...")
        print(f"\nIRAC Structure:")
        irac = result.get('irac', {})
        print(f"  - Issue: {irac.get('issue', 'N/A')[:100]}...")
        print(f"  - Rule: {irac.get('rule', 'N/A')[:100]}...")
        print(f"  - Application: {irac.get('application', 'N/A')[:100]}...")
        print(f"  - Conclusion: {irac.get('conclusion', 'N/A')[:100]}...")
        print(f"\nCitations: {result.get('citations', [])}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Open-Ended Baseline Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING B1: SINGLE-AGENT BASELINE (GROQ)")
    print("=" * 60)
    
    results = []
    
    # Test 1: API Connection
    results.append(("API Connection", test_api_connection()))
    
    if not results[0][1]:
        print("\n[FAIL] API connection failed. Fix this before continuing.")
        return
    
    # Test 2: JSON Generation
    results.append(("JSON Generation", test_json_generation()))
    
    # Test 3: MCQ Baseline
    results.append(("MCQ Baseline", test_mcq_baseline()))
    
    # Test 4: Open-Ended Baseline
    results.append(("Open-Ended Baseline", test_open_ended_baseline()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\nAll tests passed! Baseline is ready to use.")
        print("\nNext steps:")
        print("1. Run experiments: python -m src.experiments.run_baseline")
        print("2. Check results in: results/baseline_bar_exam_qa_5.json")
    else:
        print("\nSome tests failed. Check errors above.")


if __name__ == "__main__":
    main()
