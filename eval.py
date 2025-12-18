import os
import uuid
from langsmith import Client, evaluate
from langchain_groq import ChatGroq
from src.graph import graph
from dotenv import load_dotenv

load_dotenv()

# Initialize Client
client = Client()

def create_dataset():
    """Creates or Retrieves the evaluation dataset."""
    dataset_name = "Weather_RAG_Eval_Dataset"
    
    if client.has_dataset(dataset_name=dataset_name):
        return client.read_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="Dataset for evaluating Weather and RAG agent."
    )
    
    client.create_examples(
        inputs=[
            {"question": "What is the weather in Tokyo?"},
            {"question": "Tell me about the weather in New York."},
        ],
        outputs=[
            {"answer": "Weather information for Tokyo."},
            {"answer": "Weather information for New York."},
        ],
        dataset_id=dataset.id,
    )
    return dataset

def target(inputs: dict) -> dict:
    """Run the graph for a single input."""
    # Generate a random thread_id for isolation
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Invoke graph
    # Inputs from dataset match AgentState keys (question)
    result = graph.invoke(inputs, config)
    
    # Return the generated answer
    return {"answer": result.get("answer", "No answer produced")}

def run_evaluation():
    dataset_name = "Weather_RAG_Eval_Dataset"
    create_dataset()
    
    # Initialize Judge LLM
    eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # Custom Evaluator without langchain.evaluation dependency
    def correctness_eval(run, example):
        """Custom evaluator using direct LLM prompt."""
        prediction = run.outputs["answer"]
        reference = example.outputs["answer"]
        question = example.inputs["question"]
        
        system_prompt = "You are an evaluator. Grade the prediction against the reference answer. Return a score 0-1 and a brief explanation."
        user_prompt = f"Question: {question}\nReference: {reference}\nPrediction: {prediction}\n\nFormat: 'Score: <0-1> Reason: <text>'"
        
        response = eval_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]).content
        
        # Simple parser (robustness can be improved)
        try:
            import re
            score_match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", response)
            score = float(score_match.group(1)) if score_match else 0.5
        except:
            score = 0.5
            
        return {
            "key": "correctness", 
            "score": score, 
            "comment": response
        }

    # Run Evaluation
    evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_eval],
        experiment_prefix="weather-rag-eval",
        metadata={"version": "1.0.0", "llm": "llama-3.3-70b"}
    )

if __name__ == "__main__":
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Error: LANGCHAIN_API_KEY not found in environment.")
    else:
        print("Starting LangSmith evaluation...")
        run_evaluation()
