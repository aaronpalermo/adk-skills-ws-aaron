import logging
import os
from typing import Any, Dict, Optional

import requests
import vertexai
from dotenv import load_dotenv
from vertexai.preview import reasoning_engines

from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.models import Gemini, LlmRequest, LlmResponse
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("readynow_agent")

MODEL_NAME = os.getenv("MODEL", "gemini-2.5-flash")
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=30)


# 1. Callbacks for Logging
# Attached to every agent below so all user <-> agent traffic is audited,
# regardless of which specialist ends up handling a given turn.
def log_query_to_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Audit log: record the user's message before it reaches the model."""
    if llm_request.contents:
        last = llm_request.contents[-1]
        if last.role == "user" and last.parts and last.parts[0].text:
            logger.info(
                "[FEMA AUDIT] [%s] USER >> %s",
                callback_context.agent_name,
                last.parts[0].text.strip(),
            )
    return None


def log_model_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Audit log: record the model's response after it is generated."""
    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                logger.info(
                    "[FEMA AUDIT] [%s] AGENT >> %s",
                    callback_context.agent_name,
                    part.text.strip(),
                )
            elif part.function_call:
                logger.info(
                    "[FEMA AUDIT] [%s] TOOL CALL >> %s",
                    callback_context.agent_name,
                    part.function_call.name,
                )
    return None


# 2. Tool Definitions
def _log_http_error(context: str, e: Exception) -> None:
    """Logs the full HTTP response body (not just the generic exception text) so that
    permission/enablement errors from Google APIs (e.g. SERVICE_DISABLED, key
    restrictions) are visible in the audit logs without needing to reproduce them by hand."""
    response = getattr(e, "response", None)
    if response is not None:
        logger.warning("%s failed: HTTP %s -- %s", context, response.status_code, response.text[:500])
    else:
        logger.warning("%s failed: %s", context, e)


def get_lat_lon(location: str) -> Optional[Dict[str, float]]:
    """Converts a location name to latitude and longitude using the Google Maps Geocoding API."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": location, "key": api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            coords = data["results"][0]["geometry"]["location"]
            # Google returns 'lng', but we'll map it to 'lon' as requested
            return {"lat": coords["lat"], "lon": coords["lng"]}

        logger.warning(
            "Geocoding failed for %r: status=%s error_message=%s",
            location, data.get("status"), data.get("error_message"),
        )
    except Exception as e:
        _log_http_error(f"Geocoding request for {location!r}", e)
    return None


def get_weather_alerts(location: str) -> str:
    """Fetches real-time weather alerts for a specific location from the National Weather Service."""
    coords = get_lat_lon(location)
    if not coords:
        return f"Coordinate lookup failed for: {location}. Please provide a valid city or address."

    lat, lon = coords["lat"], coords["lon"]
    # NWS requires a descriptive User-Agent identifying the application.
    headers = {"User-Agent": "(FEMA-ReadyNow-POC-for-google-class)"}

    try:
        alerts_url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
        response = requests.get(alerts_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        features = data.get("features", [])
        if not features:
            return f"Currently, there are no active weather alerts for {location}."

        alert_summaries = []
        for feature in features:
            props = feature.get("properties", {})
            alert_summaries.append(
                f"WARNING: {props.get('headline')}\n{props.get('description', '')[:500]}..."
            )

        return f"Weather Alerts for {location}:\n\n" + "\n\n".join(alert_summaries)
    except Exception as e:
        _log_http_error(f"NWS alerts request for {location!r}", e)
        return f"Error connecting to weather services for {location}: {str(e)}"


def get_safety_routes(origin: str, destination: str) -> Dict[str, Any]:
    """Calculates a driving evacuation route between two locations using the Google Maps Routes API."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"error": "Google Maps API key is not configured."}

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.legs.steps.navigationInstruction",
    }
    body = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "DRIVE",
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        data = response.json()

        routes = data.get("routes")
        if not routes:
            logger.warning(
                "Routes API returned no route from %r to %r: %s", origin, destination, data
            )
            return {"error": f"No route found from {origin} to {destination}."}

        route = routes[0]
        steps = [
            step.get("navigationInstruction", {}).get("instructions", "")
            for leg in route.get("legs", [])
            for step in leg.get("steps", [])
        ]
        steps = [step for step in steps if step]

        duration_seconds = int(str(route.get("duration", "0s")).rstrip("s") or 0)
        distance_meters = route.get("distanceMeters", 0)

        return {
            "recommended_route": " -> ".join(steps) or f"Route to {destination}",
            "estimated_travel_time": f"{duration_seconds // 60} min",
            "distance": f"{distance_meters / 1609.34:.1f} miles",
        }
    except Exception as e:
        _log_http_error(f"Routes API request from {origin!r} to {destination!r}", e)
        return {"error": f"Error retrieving route from {origin} to {destination}: {str(e)}"}


def mark_session_greeted(tool_context: ToolContext) -> Dict[str, str]:
    """Records that the one-time session introduction has already been shown."""
    tool_context.state["greeted"] = True
    return {"status": "success"}


# 3. Specialized Agents
weather_agent = Agent(
    name="WeatherSpecialist",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Meteorologist providing urgent disaster-related weather updates.",
    instruction=(
        "Use the get_weather_alerts tool to check for active alerts at the user's location. "
        "Provide concise and actionable weather bulletins. Prioritize immediate threats to life and property."
    ),
    tools=[get_weather_alerts],
    output_key="weather_info",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

search_agent = Agent(
    name="SearchSpecialist",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Information Officer monitoring real-time disaster news and official sources.",
    instruction=(
        "Use Google Search to find the latest verified disaster news, official alerts, and community "
        "updates relevant to the user's question. Filter out unverified rumors and cite what you find."
    ),
    tools=[google_search],
    output_key="news_info",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

routing_agent = Agent(
    name="NavigationGuide",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Logistics Expert specializing in safe evacuation and shelter navigation.",
    instruction=(
        "Use the get_safety_routes tool to calculate a route from the user's origin to their "
        "destination (or nearest shelter). Recommend the safest evacuation path and clearly state "
        "the estimated travel time and distance."
    ),
    tools=[get_safety_routes],
    output_key="route_info",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

qa_agent = Agent(
    name="InfoOfficer",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Emergency Preparedness Expert answering general safety questions.",
    instruction=(
        "Use official FEMA safety guidelines to answer general questions about Go-Bags, first aid, "
        "and disaster preparedness."
    ),
    output_key="qa_info",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

# 4. Root Agent (Coordinator)
# Describes the system's capabilities and coordinates the specialist agents above.
#
# Each specialist is wrapped in an AgentTool rather than listed as a `sub_agents` transfer target.
# Gemini rejects any request that mixes the built-in `google_search` tool with other tools
# ("Multiple tools are supported only when they are all search tools"), and ADK automatically adds
# a `transfer_to_agent` function tool to every agent in a sub_agents transfer graph -- which would
# collide with SearchSpecialist's google_search tool the moment it became part of that graph.
# AgentTool avoids this: each specialist runs its own isolated turn (with only its own tool), and
# returns its result to the coordinator as a normal tool response, so the coordinator keeps control
# and can synthesize one combined answer.
ready_now_agent = Agent(
    name="ReadyNowRoot",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description=(
        "The central command coordinator for the FEMA ReadyNow! system. Helps people get real-time "
        "updates during a disaster: what's happening, where to go, and how to stay safe."
    ),
    instruction=(
        "SESSION GREETED: { greeted? }\n"
        "MISSION CHECK: { mission_check? }\n\n"
        "If SESSION GREETED is empty, this is the first message of the session: call your "
        "mark_session_greeted tool once, then start your reply with a brief one-time "
        "introduction -- say you're ReadyNow!, that you help with real-time disaster updates "
        "(what's happening, where to go, how to stay safe), and give 2-3 example questions "
        "people can ask you. Never repeat this introduction once SESSION GREETED is set.\n\n"
        "If MISSION CHECK starts with 'REJECT', politely relay that rejection message to the user "
        "verbatim (after the introduction, if this is the first message) and do not call any of "
        "your other tools.\n\n"
        "Otherwise, evaluate the user's situation and call the appropriate tool(s):\n"
        "- WeatherSpecialist for weather conditions and alerts\n"
        "- SearchSpecialist for internet searches on breaking disaster news\n"
        "- NavigationGuide for evacuation routes and shelter directions\n"
        "- InfoOfficer for general safety/preparedness questions\n"
        "If the user needs multiple types of help (e.g., weather AND a route), call each relevant "
        "tool and combine their findings into one clear, complete safety plan. "
        "Prioritize human safety above all."
    ),
    tools=[
        AgentTool(agent=weather_agent),
        AgentTool(agent=search_agent),
        AgentTool(agent=routing_agent),
        AgentTool(agent=qa_agent),
        mark_session_greeted,
    ],
    output_key="draft_response",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

# 5. Sequential Workflow: validate -> coordinate -> refine
guardrail_validator = Agent(
    name="Guardrail",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Safety Officer ensuring the conversation stays within FEMA ReadyNow!'s mission.",
    instruction=(
        "Evaluate the user's latest message. If it is related to disasters, emergency preparedness, "
        "weather, evacuation, shelter, or safety, respond with exactly: APPROVED\n"
        "If it is NOT related to that mission (e.g. small talk, coding help, recipes, unrelated trivia), "
        "respond with: REJECT: I'm sorry, I am currently optimized to assist strictly with emergency "
        "preparedness and disaster safety. How can I help you stay safe today?"
    ),
    output_key="mission_check",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

guardrail_refiner = Agent(
    name="ResponseRefiner",
    model=Gemini(model=MODEL_NAME, retry_options=RETRY_OPTIONS),
    description="Safety Officer ensuring final response quality and clarity.",
    instruction=(
        "MISSION CHECK: { mission_check? }\n"
        "DRAFT RESPONSE: { draft_response? }\n\n"
        "If MISSION CHECK starts with 'REJECT', output that rejection message unchanged and stop.\n"
        "Otherwise, rewrite DRAFT RESPONSE to be empathetic, clear, well-written, and easy to "
        "understand for someone who may be reading it under stress. Do not invent facts that were "
        "not present in the draft."
    ),
    output_key="final_response",
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

root_agent = SequentialAgent(
    name="ReadyNowWorkflow",
    description=(
        "Validates that the user's request is in scope, coordinates the FEMA ReadyNow! specialists, "
        "and refines the final response for clarity before it reaches the user."
    ),
    sub_agents=[guardrail_validator, ready_now_agent, guardrail_refiner],
)

# 6. Deployment Readiness
# `app` follows the ADK CLI convention (adk web / adk run) used across this repo's other examples.
app = App(
    name="ReadyNow",
    root_agent=root_agent,
)

# `agent_engine_app` is the Vertex AI Agent Engine-ready wrapper. Initializing Vertex AI requires
# real GCP credentials, which may not be present in every local dev environment, so this is guarded
# to keep `python3 agent.py` runnable as a library/import smoke test even without cloud auth.
agent_engine_app = None
try:
    vertexai.init(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    agent_engine_app = reasoning_engines.AdkApp(agent=root_agent)
except Exception as e:
    logger.warning(
        "Vertex AI Agent Engine app not initialized (expected outside a configured GCP environment): %s",
        e,
    )

# 7. Splash Screen
# `adk run` imports this module and immediately drops the user into an `[user]:` input
# prompt with no introduction, so we print a welcome banner here at module load time --
# this runs once whenever agent.py is imported, before that prompt appears (also shows
# up harmlessly in `adk web` server logs and plain `python agent.py` smoke-test output).
SAMPLE_PROMPTS = [
    "Are there any emergency alerts near College Station, TX?",
    "What's the safest evacuation route from Houston, TX to Dallas, TX?",
    "What should I pack in a Go-Bag for a hurricane?",
    "What's the latest news on the wildfire near Austin, TX?",
]

print(
    "\n" + "=" * 64 + "\n"
    "  FEMA ReadyNow! -- Emergency Preparedness Chat Agent\n"
    + "=" * 64 + "\n"
    "Hi, I'm ReadyNow! During a disaster, I can help you find out\n"
    "what's happening, where to go, and how to stay safe.\n\n"
    "Try asking me things like:\n"
    + "\n".join(f'  - "{p}"' for p in SAMPLE_PROMPTS) + "\n\n"
    "Type your question below to get started.\n"
    + "=" * 64 + "\n"
)


if __name__ == "__main__":
    # Structural smoke test: confirms every agent/tool/import above constructed successfully.
    print(f"[OK] Loaded root_agent '{root_agent.name}' with {len(root_agent.sub_agents)} workflow steps:")
    for step in root_agent.sub_agents:
        print(f"  - {step.name}: {step.description}")

    # Optional live functional smoke test, only runs if Vertex AI credentials are available.
    if agent_engine_app is not None:
        try:
            user_id = "smoke-test-user"
            session = agent_engine_app.create_session(user_id=user_id)
            test_query = "What should I bring in a Go-Bag for a hurricane?"
            last_event = None
            for event in agent_engine_app.stream_query(
                user_id=user_id, session_id=session["id"], message=test_query
            ):
                last_event = event
            print(f"\n[SMOKE TEST QUERY] {test_query}")
            print(f"[SMOKE TEST RESPONSE] {last_event}")
        except Exception as e:
            print(f"\n[INFO] Skipping live smoke-test query (no usable GCP credentials here): {e}")
    else:
        print("\n[INFO] Skipping live smoke-test query (Vertex AI was not initialized).")
