from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from src.weather import WeatherAPI
from src.rag import RAGSystem

# Initialize components
weather_api = WeatherAPI()
rag_system = RAGSystem()
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Pydantic Models
class RouterOutput(BaseModel):
    """The target destination for the user query."""
    source: Literal["weather", "rag"] = Field(
        ..., 
        description="The tool to use. 'weather' for live weather data, 'rag' for document questions."
    )

class CityExtraction(BaseModel):
    """Extraction format for city names."""
    city: str = Field(..., description="The name of the city extracted from the query.")

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    source: str

def router_node(state: AgentState) -> dict:
    """Decides whether to route to Weather or RAG."""
    query = state["question"]
    
    # Structured Output for Routing
    structured_llm = llm.with_structured_output(RouterOutput)
    
    system = "You are a router. Classify the user's query."
    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    
    try:
        result = structured_llm.invoke(messages)
        return {"source": result.source}
    except Exception:
        # Fallback if structured output fails (rare)
        return {"source": "rag"}

def weather_node(state: AgentState) -> dict:
    """Fetches weather data."""
    query = state["question"]
    
    # Structured Output for City Extraction
    structured_llm = llm.with_structured_output(CityExtraction)
    system = "Extract the city name from the query."
    
    try:
        result = structured_llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
        city = result.city
        result_text = weather_api.get_weather(city)
    except Exception:
        result_text = "Error: Could not extract city name."

    return {"context": result_text}

def rag_node(state: AgentState) -> dict:
    """Retrieves documents."""
    query = state["question"]
    docs = rag_system.retrieve(query)
    context = "\n\n".join([d.page_content for d in docs])
    return {"context": context}

def generate_node(state: AgentState) -> dict:
    """Generates an answer based on context."""
    query = state["question"]
    context = state["context"]
    source = state["source"]
    
    system = f"You are a helpful assistant. Answer the user's question based on the following context. Context Source: {source}\n\nContext:\n{context}"
    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    
    response = llm.invoke(messages)
    return {"answer": response.content}
