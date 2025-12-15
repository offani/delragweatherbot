import os
import requests
from dotenv import load_dotenv

load_dotenv()

class WeatherAPI:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city: str) -> str:
        """Fetch current weather for a given city."""
        if not self.api_key:
            return "Error: OpenWeatherMap API key not found."

        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            return (f"Weather in {city}: {weather_desc}. "
                    f"Temperature: {temp}°C (Feels like: {feels_like}°C). "
                    f"Humidity: {humidity}%. Wind Speed: {wind_speed} m/s.")
        
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"
        except KeyError:
            return f"Error: Could not parse weather data for city '{city}'."

if __name__ == "__main__":
    # Simple test
    weather = WeatherAPI()
    print(weather.get_weather("London"))
