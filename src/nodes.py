import httpx
from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from src.weather import WeatherAPI
from src.rag import RAGSystem
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize components
weather_api = WeatherAPI()
rag_system = RAGSystem()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,api_key=os.getenv("GROQ_API_KEY"),http_client=httpx.Client(verify=False),
        streaming=True)

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

from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    source: str
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Conversation history

def router_node(state: AgentState) -> dict:
    """Decides whether to route to Weather or RAG."""
    query = state["question"]
    
    # Structured Output for Routing
    structured_llm = llm.with_structured_output(RouterOutput)
    
    system = "You are a router. Classify the user's query. You have a realtime weather API and a document retrieval system (RAG). If the user is asking about current weather conditions, route to 'weather'. For all other queries, route to 'rag'. Respond ONLY with 'weather' or 'rag'."
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
    
    if not docs or len(docs) == 0:
        context = "No documents have been uploaded yet. Please upload a PDF document first to ask questions about it."
    else:
        context = "\n\n".join([d.page_content for d in docs])
    
    return {"context": context}

def generate_node(state: AgentState) -> dict:
    """Generates an answer based on context."""
    query = state["question"]
    context = state["context"]
    source = state["source"]
    conversation_messages = state.get("messages", [])
    
    system = f"""You are a helpful assistant helping user with weather and queries related to provided context. Answer the user's question based on the following context. 
    
    RULES:
1. For weather questions: ALWAYS be concise and provide accurate information.
2. For greetings (hello, hi, hey): Respond warmly and ask how you can help with weather or document queries.
3. For ANY other questions (facts, advice, etc): Politely  request to ask weather or document queries and say: "Apologies, I can help you with weather information and document queries."
4. Use the conversation history to maintain context and provide relevant responses.
5. For document queries, always refer to the provided context.

    Context Source: {source}\n\nContext:\n{context}"""
    
    messages = [SystemMessage(content=system)]
    
    # Add conversation history from checkpointer
    messages.extend(conversation_messages)
    
    # Add current query
    messages.append(HumanMessage(content=query))
    
    response = llm.invoke(messages)
    
    # Return with messages to update the checkpoint
    return {
        "answer": response.content,
        "messages": [HumanMessage(content=query), response]
    }
