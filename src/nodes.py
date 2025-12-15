from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from src.weather import WeatherAPI
from src.rag import RAGSystem

# Initialize components
weather_api = WeatherAPI()
rag_system = RAGSystem()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    source: str

def router_node(state: AgentState) -> dict:
    """Decides whether to route to Weather or RAG."""
    query = state["question"]
    
    # Simple router using LLM
    system = "You are a router. Your job is to classify the user's query into one of two categories: 'WEATHER' or 'RAG'. 'WEATHER' is for queries about current weather conditions in specific locations. 'RAG' is for general questions or questions about documents. Return ONLY the category name."
    messages = [SystemMessage(content=system), HumanMessage(content=query)]
    
    response = llm.invoke(messages).content.strip().upper()
    
    # Fallback/Safety
    if "WEATHER" in response:
        return {"source": "weather"}
    else:
        return {"source": "rag"}

def weather_node(state: AgentState) -> dict:
    """Fetches weather data."""
    query = state["question"]
    # Extract city using LLM for better accuracy
    system = "Extract the city name from the query. Return ONLY the city name."
    city = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)]).content.strip()
    
    result = weather_api.get_weather(city)
    return {"context": result}

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
