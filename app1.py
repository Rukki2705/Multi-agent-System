import time
import random
import pandas as pd
import streamlit as st
import requests
from typing import Optional
from groq import Groq
from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_groq import ChatGroq
import os

# ----------------------------
# 1. Internal Functions
# ----------------------------
def _get_building_summary_internal() -> str:
    df = pd.read_csv("update.csv")
    df.columns = [col.strip() for col in df.columns]
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)
    return (
        f"Average Temperature: {df['Temperature'].mean():.2f} °C\n"
        f"Average CO₂ Level: {df['CO2'].mean():.2f} ppm\n"
        f"Average Light Level: {df['Light'].mean():.2f} Lux\n"
        f"Average Humidity: {df['Humidity'].mean():.2f} %\n"
        f"Average Occupancy: {df['Occupancy'].mean():.2f}"
    )

def _get_weather_summary_internal(city: str) -> str:
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_resp = requests.get(geo_url).json()
        results = geo_resp.get("results", [])
        if not results:
            return f"City '{city}' not found."

        lat = results[0]['latitude']
        lon = results[0]['longitude']

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m&timezone=auto"
        weather_resp = requests.get(weather_url).json()
        data = weather_resp.get("current", {})

        return (
            f"Weather for {city} (lat: {lat}, lon: {lon}):\n"
            f"Outside Temp: {data.get('temperature_2m', 'N/A')} °C\n"
            f"Humidity: {data.get('relative_humidity_2m', 'N/A')}%\n"
            f"Cloud Cover: {data.get('cloud_cover', 'N/A')}%\n"
            f"Wind Speed: {data.get('wind_speed_10m', 'N/A')} km/h"
        )
    except Exception as e:
        return f"Error fetching weather data: {e}"

def call_groq_fallback(prompt: str, api_key: str, model: str, label: str) -> str:
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a focused assistant specialized in {label}. Do not provide overlapping content from other domains."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[{label}] Groq fallback failed: {e}"

# ----------------------------
# 2. CrewAI Tool Wrappers
# ----------------------------
@tool
def get_building_summary() -> str:
    """Return average environmental and occupancy statistics for the building."""
    return _get_building_summary_internal()

@tool
def get_weather_summary(city: str) -> str:
    """Return current weather conditions for a given city using Open-Meteo API."""
    return _get_weather_summary_internal(city)

# ----------------------------
# 3. Live Room Data Generator
# ----------------------------
def get_live_room_data(num_rooms: int = 5) -> pd.DataFrame:
    rooms = []
    room_numbers = random.sample(range(1, 101), num_rooms)
    for room_num in room_numbers:
        light = random.randint(100, 600)
        lighting_status = random.choice(["Full", "Dimmed", "Off"])
        temperature = round(random.uniform(18.0, 26.0), 1)
        hvac_status = random.choice(["Heating", "Cooling", "Off"])

        if hvac_status == "Heating" and temperature > 24:
            hvac_action = f"Turn off heating in Room_{room_num}"
        elif hvac_status == "Cooling" and temperature < 20:
            hvac_action = f"Turn off cooling in Room_{room_num}"
        elif hvac_status == "Off" and (temperature < 18 or temperature > 26):
            hvac_action = f"Enable HVAC in Room_{room_num} for comfort"
        else:
            hvac_action = f"Maintain Room_{room_num} HVAC"

        if lighting_status != "Off" and light > 400:
            lighting_action = f"Turn off lights in Room_{room_num}"
        elif lighting_status == "Full" and light > 250:
            lighting_action = f"Dim lights in Room_{room_num}"
        else:
            lighting_action = f"Maintain Room_{room_num} lighting"

        rooms.append({
            "RoomID": f"Room_{room_num}",
            "Occupancy": round(random.uniform(0.0, 1.0), 2),
            "Temperature": temperature,
            "CO2": random.randint(400, 1200),
            "Light": light,
            "Humidity": round(random.uniform(20.0, 60.0), 1),
            "Power_kWh": round(random.uniform(0.2, 3.5), 2),
            "HVAC_Status": hvac_status,
            "Lighting_Status": lighting_status,
            "Auto_Lighting_Action": lighting_action,
            "Auto_HVAC_Action": hvac_action
        })
    return pd.DataFrame(rooms)

# ----------------------------
# 4. Streamlit App with Chat History
# ----------------------------
st.set_page_config(page_title="CrewAI Smart Space & Energy Optimization", layout="wide")

st.sidebar.title("🔧 Configuration Panel")
groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
num_rooms = st.sidebar.slider("🏠 Number of Rooms", min_value=3, max_value=15, value=5, step=1)
city_name = st.sidebar.text_input("🌆 Enter City for Weather", value="New York")
agent_selection = st.sidebar.selectbox("🧠 Select Agent Task", ["Select Agent", "Space Optimization", "Energy Optimization"])

os.environ["GROQ_API_KEY"] = groq_api_key
model_choice = "llama-3.3-70b-versatile"

if groq_api_key and agent_selection != "Select Agent":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    building_summary = _get_building_summary_internal()
    weather_summary = _get_weather_summary_internal(city_name)

    if "rooms_df" not in st.session_state or st.session_state.get("num_rooms") != num_rooms:
        st.session_state.rooms_df = get_live_room_data(num_rooms=num_rooms)
        st.session_state.num_rooms = num_rooms

    rooms_df = st.session_state.rooms_df
    rooms_snapshot = "\n".join([
        f"Room {row['RoomID']}: Occupancy={row['Occupancy']}, Temp={row['Temperature']}°C, CO₂={row['CO2']}ppm, Light={row['Light']} Lux, Humidity={row['Humidity']}%, Power={row['Power_kWh']} kWh, HVAC={row['HVAC_Status']} -> {row['Auto_HVAC_Action']}, Lighting={row['Lighting_Status']} -> {row['Auto_Lighting_Action']}"
        for _, row in rooms_df.iterrows()
    ])

    groq_llm = ChatGroq(model=model_choice, temperature=0.3)

    st.subheader("📋 Live Room Snapshot")
    st.dataframe(rooms_df)

    if agent_selection == "Space Optimization":
        st.subheader("🏢 Building Summary")
        st.text(building_summary)
        st.markdown("### 💬 Conversation with Space Optimization Agent")
    else:
        st.subheader("🌤️ Weather Summary")
        st.text(weather_summary)
        st.markdown("### 💬 Conversation with Energy Optimization Agent")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("How can I help you?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if agent_selection == "Space Optimization":
            agent = Agent(
                role="Space Optimization Assistant",
                goal="Optimize room allocation using occupancy and environment data.",
                backstory="You're an expert in spatial optimization for sustainable building operations.",
                verbose=True,
                allow_delegation=False,
                tools=[get_building_summary],
                llm=groq_llm
            )
            task = Task(description=user_input, expected_output="Room reallocation suggestions only.", agent=agent)
        else:
            agent = Agent(
                role="Energy Optimization Assistant",
                goal="Provide recommendations on HVAC, lighting, and energy saving strategies.",
                backstory="You're responsible for optimizing building energy systems while maintaining comfort.",
                verbose=True,
                allow_delegation=False,
                tools=[get_weather_summary],
                llm=groq_llm
            )
            task = Task(description=user_input, expected_output="Concise energy-saving recommendations.", agent=agent)

        try:
            output = Crew(agents=[agent], tasks=[task]).kickoff()
        except Exception as e:
            context = building_summary if agent_selection == "Space Optimization" else weather_summary
            prompt = f"You are an assistant.\nContext:\n{context}\n{rooms_snapshot}\nUser: {user_input}"
            output = call_groq_fallback(prompt, groq_api_key, model_choice, agent_selection.replace(" ", ""))

        st.session_state.messages.append({"role": "assistant", "content": output})
        with st.chat_message("assistant"):
            st.markdown(output)
else:
    st.info("Please enter a valid API key and select an agent task from the sidebar to begin.")
