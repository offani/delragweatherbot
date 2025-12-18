import pytest
from unittest.mock import MagicMock, patch
from src.weather import WeatherAPI
from src.nodes import router_node, weather_node
from src.rag import RAGSystem

# Mock env vars
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "fake_key")
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")

def test_weather_api_success():
    with patch("src.weather.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "weather": [{"description": "sunny"}],
            "main": {"temp": 25, "feels_like": 27, "humidity": 60},
            "wind": {"speed": 5}
        }
        mock_get.return_value = mock_response
        
        weather = WeatherAPI()
        result = weather.get_weather("London")
        assert "Weather in London: sunny" in result
        assert "Temperature: 25°C" in result


# Mock Pydantic models
class MockRouterOutput:
    def __init__(self, source):
        self.source = source

class MockCityExtraction:
    def __init__(self, city):
        self.city = city

def test_router_node_weather():
    # Mock LLM to return "WEATHER"
    with patch("src.nodes.llm") as mock_llm:
        mock_llm.with_structured_output.return_value.invoke.return_value = MockRouterOutput(source="weather")
        state = {"question": "What's the weather in Paris?", "context": "", "answer": "", "source": ""}
        result = router_node(state)
        assert result["source"] == "weather"

def test_router_node_rag():
    with patch("src.nodes.llm") as mock_llm:
        mock_llm.with_structured_output.return_value.invoke.return_value = MockRouterOutput(source="rag")
        state = {"question": "Summarize the document.", "context": "", "answer": "", "source": ""}
        result = router_node(state)
        assert result["source"] == "rag"

def test_weather_node():
    with patch("src.nodes.llm") as mock_llm:
        # Mock city extraction
        mock_llm.with_structured_output.return_value.invoke.return_value = MockCityExtraction(city="London")
        
        with patch("src.nodes.weather_api.get_weather") as mock_weather:
            mock_weather.return_value = "Sunny in London"
            state = {"question": "Weather in London", "context": "", "answer": "", "source": "weather"}
            result = weather_node(state)
            assert result["context"] == "Sunny in London"


