import os
import sys
import logging

from dotenv import load_dotenv
import google.cloud.logging
from google.adk import Agent
from google.adk.models import Gemini
from google.genai import types
from typing import Optional, List, Dict
import requests

from google.adk.tools.tool_context import ToolContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from google.adk.apps.app import App

import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest


def log_query_to_model(callback_context: CallbackContext, llm_request: LlmRequest):
    if llm_request.contents and llm_request.contents[-1].role == 'user':
        for part in llm_request.contents[-1].parts:
            if part.text:
                logging.info("[query to %s]: %s", callback_context.agent_name, part.text)

def log_model_response(callback_context: CallbackContext, llm_response: LlmResponse):
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                logging.info("[response from %s]: %s", callback_context.agent_name, part.text)
            elif part.function_call:
                logging.info("[function call from %s]: %s", callback_context.agent_name, part.function_call.name)



load_dotenv()

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=30)
_MODEL_ARMOR_LOCATION = os.getenv("MODEL_ARMOR_LOCATION", "us-central1")
_MODEL_ARMOR_TEMPLATE = os.getenv("MODEL_ARMOR_TEMPLATE", "safety-dance")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

def get_gov_weather_forecast(lat: float, lon: float) -> Optional[List[Dict[str, str]]]:
    """
    Fetch today's weather forecast from the U.S. National Weather Service
    API based on a given latitude and longitude.

    Args:
        lat (float): Latitude of the location (e.g., 38.8977).
        lon (float): Longitude of the location (e.g., -77.0365).

    Returns:
        Optional[List[Dict[str, str]]]: A list of forecast period dictionaries,
        each with 'name', 'temperature', 'shortForecast', and 'detailedForecast'.
        Returns None if data is unavailable (including for non-US locations,
        which NWS does not cover) or an error occurs.
    """
    # NWS requires a descriptive User-Agent identifying the application.
    headers = {"User-Agent": "(google-lab-readynow-weather-agent)"}

    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_resp = requests.get(points_url, headers=headers, timeout=10)
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecast"]

        forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json()["properties"]["periods"]

        return [
            {
                "name": period["name"],
                "temperature": f"{period['temperature']}°{period['temperatureUnit']}",
                "shortForecast": period["shortForecast"],
                # "detailedForecast": period["detailedForecast"],
            }
            for period in periods
        ]
    except (requests.RequestException, KeyError):
        return None

def get_lat_lon(location: str) -> Optional[Dict[str, float]]:
    """
    Convert a place name or address into latitude and longitude using the
    Google Maps Geocoding API.

    Args:
        location (str): A place name, city, or address (e.g., "College Station, TX").

    Returns:
        Optional[Dict[str, float]]: A dictionary with 'lat' and 'lon' keys.
        Returns None if the location cannot be found or an error occurs.
    """
    # breakpoint()
    if not GOOGLE_MAPS_API_KEY:
        return "google api key not found"

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": GOOGLE_MAPS_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK" or not data.get("results"):
            return None

        coords = data["results"][0]["geometry"]["location"]
        return {"lat": coords["lat"], "lon": coords["lng"]}
    except (requests.RequestException, KeyError, IndexError):
        return "exception, try again"

# Agents

weather_checker = Agent(
    name="weather_checker",
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    description="Build a list of attractions to visit in a city based on today's weather.",
    instruction="""
        - Provide the user options for attractions to visit within their selected city based on today's weather.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    # When instructed to do so, paste the tools parameter below this line
    # tools=[save_attractions_to_state]
    tools=[get_lat_lon, get_gov_weather_forecast]
    )

fun_finder = Agent(
    name="fun_finder",
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    description="Help a user decide what city in the USA to visit.",
    instruction="""
        Provide a few suggestions of popular cities in the USA for travelers.
        
        Help a user identify their primary goals of travel:
        adventure, leisure, learning, shopping, or viewing art

        Identify cities that would make great destinations
        based on their priorities.
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

root_agent = Agent(
    name="steering",
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    description="Start a user on a travel adventure.",
    instruction="""
        Ask the user if they know where they'd like to travel
        or if they need some help deciding.
        If they need help deciding, send them to
        'fun_finder'.
        If they know what city they'd like to visit,
        send them to the 'weather_checker'.
        """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    # Add the sub_agents parameter when instructed below this line
    sub_agents=[fun_finder, weather_checker]
)


app = App(
    name="fun_finder",
    root_agent=root_agent,
)
