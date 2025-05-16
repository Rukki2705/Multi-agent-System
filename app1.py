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
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# ----------------------------
# 1. Internal Functions
# ----------------------------
def _get_building_summary_internal() -> str:
    df = pd.read_csv("update.csv")
    df.columns = [col.strip() for col in df.columns]
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)
    return (
        f"Average Temperature: {df['Temperature'].mean():.2f} °C\\n"
        f"Average CO₂ Level: {df['CO2'].mean():.2f} ppm\\n"
        f"Average Light Level: {df['Light'].mean():.2f} Lux\\n"
        f"Average Humidity: {df['Humidity'].mean():.2f} %\\n"
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
            f"Weather for {city} (lat: {lat}, lon: {lon}):\\n"
            f"Outside Temp: {data.get('temperature_2m', 'N/A')} °C\\n"
            f"Humidity: {data.get('relative_humidity_2m', 'N/A')}%\\n"
            f"Cloud Cover: {data.get('cloud_cover', 'N/A')}%\\n"
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

@tool
def predict_occupancy_trend(room_id: Optional[str] = None) -> str:
    """Forecast future occupancy for a given room using ARIMA. Defaults to all rooms if not specified."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        room_ids = [room_id] if room_id else rooms_df["RoomID"].unique()
        forecast_results = []

        now = pd.Timestamp.now()

        for rid in room_ids:
            # Generate synthetic 48-point historical occupancy data (last 48 hours)
            history = [
                round(random.uniform(0.0, 1.0), 2)
                for _ in range(48)
            ]

            try:
                model = ARIMA(history, order=(2, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=6)  # next 6 hours
                forecast_values = [round(val, 2) for val in forecast.tolist()]

                avg = sum(forecast_values) / len(forecast_values)
                max_occ = max(forecast_values)
                min_occ = min(forecast_values)

                forecast_results.append(
                    f"📊 Occupancy forecast for {rid} (next 6 hours):\n"
                    f"• Avg: {avg:.2f}, Max: {max_occ:.2f}, Min: {min_occ:.2f}\n"
                    f"• Trend: {', '.join(map(str, forecast_values))}"
                )
            except:
                forecast_results.append(f"⚠️ Forecast model failed for {rid}.")

        return "\n\n".join(forecast_results)

    except Exception as e:
        return f"❌ Occupancy trend forecast failed: {e}"

@tool
def simulate_energy_reduction(room: dict) -> str:
    """Simulate energy reduction by optimizing HVAC and lighting settings based on model predictions."""
    try:
        model = joblib.load("energy_model.pkl")
    except Exception as e:
        return f"❌ Failed to load model: {e}"

    current_df = pd.DataFrame([room])
    current_df_encoded = pd.get_dummies(current_df)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in current_df_encoded.columns:
            current_df_encoded[col] = 0
    current_df_encoded = current_df_encoded[model_features]
    current_pred = model.predict(current_df_encoded)[0]

    optimized = room.copy()
    optimized["Lighting_Status"] = "Dimmed"
    optimized["HVAC_Status"] = "Off"
    optimized["Light"] = min(room["Light"], 200)
    optimized["Temperature"] = min(max(room["Temperature"], 22), 24)

    optimized_df = pd.DataFrame([optimized])
    optimized_df_encoded = pd.get_dummies(optimized_df)
    for col in model_features:
        if col not in optimized_df_encoded.columns:
            optimized_df_encoded[col] = 0
    optimized_df_encoded = optimized_df_encoded[model_features]
    optimized_pred = model.predict(optimized_df_encoded)[0]

    reduction = current_pred - optimized_pred
    percent = (reduction / current_pred) * 100

    return (
        f"🔋 Current Energy Usage: {current_pred:.2f} kWh\\n"
        f"⚙️  Optimized Energy Usage: {optimized_pred:.2f} kWh\\n"
        f"📉 Estimated Reduction: {reduction:.2f} kWh ({percent:.1f}%)"
    )

@tool
def suggest_room_consolidation() -> str:
    """Suggest consolidation of underutilized rooms based on occupancy threshold."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        underused = rooms_df[rooms_df["Occupancy"] < 0.3]
        if underused.empty:
            return "✅ All rooms have adequate occupancy. No consolidation needed."

        suggestions = []
        for _, row in underused.iterrows():
            suggestions.append(
                f"🔄 Room {row['RoomID']} has low occupancy ({row['Occupancy']*100:.1f}%). Consider merging or closing."
            )
        return "\\n".join(suggestions)
    except Exception as e:
        return f"❌ Error during consolidation suggestion: {e}"

@tool
def suggest_space_rezoning() -> str:
    '''Cluster rooms into zones based on occupancy, temperature, CO2, light, humidity, and energy usage.'''
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        features = ["Occupancy", "Temperature", "CO2", "Light", "Humidity", "Power_kWh"]
        data = rooms_df[features]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        rooms_df["Zone"] = kmeans.fit_predict(X_scaled)
        st.session_state["rooms_df"] = rooms_df  # update stored df

        # Format result
        cluster_counts = rooms_df["Zone"].value_counts().sort_index()
        summary = "\\n".join([f"Zone {z}: {count} room(s)" for z, count in cluster_counts.items()])
        return f"🧠 Re-zoning complete using KMeans clustering:\\n{summary}"
    except Exception as e:
        return f"❌ Error during re-zoning: {e}"
    
@tool
def forecast_energy_trend(room_id: Optional[str] = None) -> str:
    """Generate a textual summary forecast of energy usage for specific or all rooms using ARIMA."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty:
            return "⚠️ No live room data available."

        now = pd.Timestamp.now()
        room_ids = [room_id] if room_id else rooms_df["RoomID"].unique()
        summaries = []

        for room in room_ids:
            # Simulate synthetic historical data
            history = [
                round(random.uniform(0.8, 2.5) + 0.3 * (0.5 - random.random()), 2)
                for _ in range(48)
            ]
            try:
                model = ARIMA(history, order=(2, 1, 2))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=12)
                forecast_values = forecast.tolist()

                avg = sum(forecast_values) / len(forecast_values)
                low = min(forecast_values)
                high = max(forecast_values)

                summaries.append(
                    f"🔮 Forecast for {room}: Avg = {avg:.2f} kWh, Range = {low:.2f} - {high:.2f} kWh over next 12 hours."
                )
            except:
                summaries.append(f"⚠️ Forecast failed for {room}.")

        return "\n".join(summaries)

    except Exception as e:
        return f"❌ Forecasting failed: {e}"

@tool
def explain_energy_forecast_llm(room_id: str) -> str:
    """LLM-based forecast summary for a room using current room and weather data."""
    try:
        rooms_df = st.session_state.get("rooms_df")
        if rooms_df is None or rooms_df.empty or room_id not in rooms_df["RoomID"].values:
            return f"⚠️ Room {room_id} not found in current data."

        room = rooms_df[rooms_df["RoomID"] == room_id].iloc[0].to_dict()
        weather = _get_weather_summary_internal(st.session_state.get("city_name", "New York"))

        prompt = (
            f"Given the following room and outside weather conditions, generate a detailed energy usage forecast.\n\n"
            f"Room: {room_id}\n"
            f"Occupancy: {room['Occupancy']}\n"
            f"Temperature: {room['Temperature']}°C\n"
            f"CO₂: {room['CO2']}ppm\n"
            f"Light: {room['Light']} Lux\n"
            f"Humidity: {room['Humidity']}%\n"
            f"Power: {room['Power_kWh']} kWh\n"
            f"HVAC: {room['HVAC_Status']}\n"
            f"Lighting: {room['Lighting_Status']}\n\n"
            f"Outside Weather:\n{weather}\n\n"
            f"Please explain:\n"
            f"1. Heating or cooling demand\n"
            f"2. Lighting adjustments and their effect\n"
            f"3. Occupancy impact\n"
            f"4. Forecasted energy usage (range) for the next few hours\n"
            f"5. Actionable recommendations"
        )

        model = ChatGroq(model=st.session_state.get("model_choice", "llama-3.3-70b-versatile"), temperature=0.3)
        response = model.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f"❌ Energy forecast failed: {e}"
    
@tool
def recommend_layout_plan() -> str:
    """Suggest layout changes based on occupancy and usage patterns."""
    try:
        df = st.session_state.get("rooms_df")
        if df is None or df.empty:
            return "⚠️ No live room data available."

        underused = df[df["Occupancy"] < 0.2]
        overused = df[df["Occupancy"] > 0.75]
        suggestions = []

        for _, row in underused.iterrows():
            suggestions.append(
                f"🔄 Room {row['RoomID']} is underused (Occupancy {row['Occupancy']:.2f}). Consider repurposing it as a quiet zone, storage, or merging."
            )

        for _, row in overused.iterrows():
            suggestions.append(
                f"📈 Room {row['RoomID']} is heavily used (Occupancy {row['Occupancy']:.2f}). Consider creating more spaces like this or redistributing usage."
            )

        return "\n".join(suggestions) if suggestions else "✅ Current layout appears optimal."

    except Exception as e:
        return f"❌ Failed to generate layout plan: {e}"

@tool
def identify_multiuse_zones() -> str:
    """Identify rooms suitable for multi-functional use based on usage and environment."""
    try:
        df = st.session_state.get("rooms_df")
        if df is None or df.empty:
            return "⚠️ No room data available."

        flexible = df[
            (df["Occupancy"] > 0.3) &
            (df["Power_kWh"] < 2.5) &
            (df["Light"] > 200) &
            (df["Humidity"] < 50)
        ]

        if flexible.empty:
            return "No rooms currently suitable for multi-functional use."

        return "\n".join([
            f"🌀 Room {row['RoomID']} could serve multiple purposes (e.g., meeting + focus work)."
            for _, row in flexible.iterrows()
        ])

    except Exception as e:
        return f"❌ Error identifying flexible rooms: {e}"


@tool
def recommend_room_function_map() -> str:
    """
    Recommend a function for each room based on occupancy, power usage, and lighting.
    Logic:
    - High occupancy + high power → Meeting Room
    - Low occupancy + high light → Focus Pod
    - Low occupancy + low power → Storage/Flex Space
    """
    try:
        df = st.session_state.get("rooms_df")
        if df is None or df.empty:
            return "⚠️ No room data available."

        suggestions = []
        for _, row in df.iterrows():
            room_id = row["RoomID"]
            occ = row["Occupancy"]
            power = row["Power_kWh"]
            light = row["Light"]

            if occ > 0.6 and power > 2.5:
                function = "🧑‍💼 Meeting Room"
            elif occ < 0.3 and light > 300:
                function = "🔕 Focus Pod"
            elif occ < 0.3 and power < 1.5:
                function = "📦 Storage or Flex Space"
            else:
                function = "🔄 Multi-use Room"

            suggestions.append(f"{room_id}: Recommended use → {function}")

        return "\n".join(suggestions)

    except Exception as e:
        return f"❌ Failed to generate room function map: {e}"


# ----------------------------
# 3. Room Generator
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

st.markdown("""
    <style>
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #1E88E5 !important;
        color: white;
    }

    /* Sidebar header and labels */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background-color: white !important;
        color: black !important;
        border-radius: 5px;
    }

    section[data-testid="stSidebar"] .stSelectbox div {
        color: black !important;
    }

    section[data-testid="stSidebar"] .stSlider .css-1t42a27,
    section[data-testid="stSidebar"] .stSlider .css-14xtw13 {
        color: white !important;
        font-weight: bold;
    }

    section[data-testid="stSidebar"] .rc-slider-track {
        background-color: white !important;
    }
    section[data-testid="stSidebar"] .rc-slider-rail {
        background-color: #90CAF9 !important;
    }
    section[data-testid="stSidebar"] .rc-slider-handle {
        background-color: #FFEB3B !important;
        border: 2px solid white !important;
    }

    section[data-testid="stSidebar"] .stButton>button {
        background-color: #43A047 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
        border: none;
    }

    .chat-container {
        display: flex;
        margin-bottom: 12px;
        align-items: flex-start;
    }
    .left .bubble {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 10px;
        max-width: 70%;
        margin-left: 10px;
    }
    .right {
        flex-direction: row-reverse;
    }
    .right .bubble {
        background-color: #ECEFF1;
        border-radius: 10px;
        padding: 10px;
        max-width: 70%;
        margin-right: 10px;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 10px;
    }
    .bubble {
        font-size: 15px;
        line-height: 1.4;
    }
    .block-container {
        padding-top: 3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🛠 Configuration Panel")
    groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    model_choice = st.selectbox(
    "🧠 Select LLM Model",
    options=[
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "mistral-saba-24b"
    ],
    index=0
    )

    num_rooms = st.slider("🏠 Number of Rooms", min_value=3, max_value=15, value=5)
    city_name = st.text_input("🌆 Enter City for Weather", value="New York")
    agent_selection = st.selectbox("🤖 Select Agent", ["Select Agent", "Space Optimization", "Energy Optimization", "Layout Recommendation"])
    if st.button("Clear/Reset"):
        st.session_state.clear()

st.markdown("<h2 style='text-align: center; color: #1565C0;'>Multi-Agent System for Energy Efficiency</h2>", unsafe_allow_html=True)

os.environ["GROQ_API_KEY"] = groq_api_key
#model_choice = "llama-3.3-70b-versatile"

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

    st.subheader("📋 Live Room Snapshot")
    st.dataframe(rooms_df, use_container_width=True)

    st.markdown("### 💬 Agent Chat Interface")

    user_input = st.chat_input("Ask the agent...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        groq_llm = ChatGroq(model=model_choice, temperature=0.3)

        if agent_selection == "Space Optimization":
            agent = Agent(
                role="Space Optimization Assistant",
                goal="Optimize room allocation using occupancy and environment data.",
                backstory="You're an expert in spatial optimization for sustainable building operations.",
                verbose=True,
                allow_delegation=False,
                tools=[get_building_summary, suggest_room_consolidation, suggest_space_rezoning, predict_occupancy_trend],
                llm=groq_llm
            )
        elif agent_selection == "Layout Recommendation":
            agent = Agent(
                role="Layout Planner",
                goal="Analyze room data and suggest long-term layout changes for flexibility and efficiency.",
                backstory="You specialize in spatial design and dynamic reconfiguration based on usage metrics.",
                verbose=True,
                allow_delegation=False,
                tools=[get_building_summary, recommend_layout_plan, identify_multiuse_zones, recommend_room_function_map],
                llm=groq_llm
            )
    
        else:
            agent = Agent(
                role="Energy Optimization Assistant",
                goal="Provide recommendations on HVAC, lighting, and energy saving strategies.",
                backstory="You're responsible for optimizing building energy systems while maintaining comfort.",
                verbose=True,
                allow_delegation=False,
                tools=[get_weather_summary, simulate_energy_reduction, forecast_energy_trend, explain_energy_forecast_llm],
                llm=groq_llm
            )

        try:
            task = Task(
                    description=user_input,
                    expected_output="Concise energy-saving recommendations.",
                    agent=agent
            )
            output = Crew(agents=[agent], tasks=[task]).kickoff()

        except Exception as e:
            context = building_summary if agent_selection == "Space Optimization" else weather_summary
            fallback_prompt = (
                f"You are an assistant.\n"
                f"Context:\n{context}\n"
                f"Room Info:\n{rooms_snapshot}\n"
                f"User: {user_input}"
            )
            try:
                output = call_groq_fallback(fallback_prompt, groq_api_key, model_choice, agent_selection.replace(" ", ""))
            except Exception as fallback_error:
                output = f"❌ Final fallback failed: {fallback_error}"


        st.session_state.messages.append({"role": "assistant", "content": output})

    # ✅ Now render chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar_url = "https://img.icons8.com/fluency/48/bot.png" if role == "assistant" else "https://img.icons8.com/fluency/48/user-male-circle.png"
        alignment = "left" if role == "assistant" else "right"

        st.markdown(f"""
            <div class="chat-container {alignment}">
                <img src="{avatar_url}" class="avatar" />
                <div class="bubble">{msg['content']}</div>
            </div>
        """, unsafe_allow_html=True)

else:
    st.info("Please enter a valid API key and select an agent task from the sidebar to begin.")


