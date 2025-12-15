import os
from langsmith import Client
from langsmith import RunEvalConfig, run_on_dataset
from langchain_groq import ChatGroq
from src.graph import graph

# Initialize LangSmith Client
client = Client()

def create_dataset():
    """Creates a sample dataset for evaluation."""
    dataset_name = "Weather_RAG_Eval_Dataset"
    
    # Check if dataset exists
    if client.has_dataset(dataset_name=dataset_name):
        return client.read_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="Dataset for evaluating Weather and RAG agent."
    )
    
    # Add examples
    client.create_examples(
        inputs=[
            {"question": "What is the weather in Tokyo?"},
            {"question": "Tell me about the weather in New York."},
        ],
        outputs=[
            {"answer": "Weather information for Tokyo."}, # Rough expectation
            {"answer": "Weather information for New York."},
        ],
        dataset_id=dataset.id,
    )
    
    return dataset

def run_evaluation():
    """Runs evaluation on the dataset."""
    dataset_name = "Weather_RAG_Eval_Dataset"
    create_dataset() # Ensure dataset exists
    
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    eval_config = RunEvalConfig(
        evaluators=[
            # Evaluates if the response is helpful and relevant
            RunEvalConfig.Criteria("helpfulness"),
            RunEvalConfig.Criteria("relevance"),
        ],
        eval_llm=llm
    )
    
    # Wrapper for graph to match expected signature if needed
    # graph.invoke takes a dict, returns a dict. 
    def graph_predict(inputs):
        return graph.invoke(inputs)

    run_on_dataset(
        client=client,
        dataset_name=dataset_name,
        llm_or_chain_factory=graph_predict,
        evaluation=eval_config,
    )

if __name__ == "__main__":
    # Ensure API keys are set
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("LangSmith API Key missing.")
    else:
        run_evaluation()
