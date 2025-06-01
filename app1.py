
import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
from pythermalcomfort.models import pmv_ppd_iso, utci
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from datetime import datetime
import random
import re
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Optional
from sklearn.linear_model import LinearRegression
import joblib
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
os.environ['SERPER_API_KEY'] = "3a39cbda3cf120c2c32e54d0c7f8d6f3cf3d78a7"
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
import re
import plotly.graph_objects as go





st.markdown("""
<style>
/* Make sidebar background and labels white */
[data-testid=stSidebar] {
  background-color: #4d85e5;
}

[data-testid=stSidebar] .stSelectbox label, 
[data-testid=stSidebar] .stMultiSelect label,
[data-testid=stSidebar] h1, 
[data-testid=stSidebar] h2, 
[data-testid=stSidebar] h3, 
[data-testid=stSidebar] h4,
[data-testid=stSidebar] p {
  color: white !important;
  font-weight: bold;
}

/* Make text inside dropdown/selectbox black */
[data-testid=stSidebar] .stSelectbox div[data-baseweb="select"] span,
[data-testid=stSidebar] .stMultiSelect div[data-baseweb="select"] span {
  color: black !important;
}

/* Make dropdown menu text black */
div[data-testid="stSidebar"] div[role="listbox"] ul li {
  color: black !important;
}

/* Style for the button text */
[data-testid=stSidebar] button {
  color: white !important;
  font-weight: bold;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<div style='text-align: center; margin-bottom: 12px;'>
    <img src="https://raw.githubusercontent.com/LLM-AI-INDIA/GenAI-Bootcamp-FEB2025/main/Lab-4/image/default_logo.png"
         alt="App Logo"
         style="width:300px; height:150px; border-radius:10%;" />
</div>
<div style='text-align: center; color: #003366; font-size: 23px; font-weight: 500; margin-bottom: 12px;'>
    Our Multi-Agent Orchestration simplifies smart building maintenance
</div>
<div style='text-align: center; color: blue; font-size: 16px; margin-top: -10px;'>
    The agentic approach saves time, reduce costs, improves accuracy and enhances user experience
</div>
<hr style="border: 1px solid gray; height:2.5px; margin-top:0px; width:100%; background-color:gray;">
""", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
openai_api_key = st.text_input("Enter your OpenAI API key:", type="password", value=st.session_state.get('openai_api_key', ''))
st.session_state['openai_api_key'] = openai_api_key
st.markdown("</div>", unsafe_allow_html=True)

# Set the API key as an environment variable
if openai_api_key:
    import os
    os.environ['OPENAI_API_KEY'] = openai_api_key

# Add after the header section and before the API configuration
st.markdown("<div style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# First table: Agent Configuration
#st.subheader("Agent Configuration")

# Create a static dataframe for the agent tools
tools_df = pd.DataFrame({
    "Tool": ["Thermal Comfort Calculator", "Energy Forecasting", "Space Analytics", "Layout Recommendation"],
    "Input": ["Indoor temperature, humidity, air speed, activity level, clothing insulation", "Power usage, HVAC status, lighting/appliance data, energy tariff", "Room-level data: occupancy, temperature, CO₂, light, humidity, energy usage", "Room usage metrics: power, occupancy, light, humidity, function patterns"],
    "Output": ["PMV, PPD, UTCI scores with thermal comfort classification", "Energy optimization actions with estimated savings & ROI", "Zone classification, underutilization alerts, consolidation suggestions", "Layout adjustments, multi-use zone identification, functional mapping"]
})

tools_df_public_sector = pd.DataFrame({
    "Use Case": [
        "Unemployment Policy",
        "Healthcare Policy"
    ],
    "Input": [
        "Region, job title, user profile (skills, experience, education)",
        "User profile (age, income, family size, insurance), CMS/state websites"
    ],
    "Output": [
        "In-demand job roles, unemployment disparities, retraining advice",
        "Matched healthcare programs and enrollment actions"
    ]
})

# Display the tools table
#st.table(tools_df)

# Second table: Agent Technology Stack


st.markdown("</div>", unsafe_allow_html=True)



# Initialize session state for storing data between agents
if 'thermal_data' not in st.session_state:
    st.session_state.thermal_data = None
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'results' not in st.session_state:
    st.session_state.results = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rooms_df" not in st.session_state:
    st.session_state.rooms_df = None
if "num_rooms" not in st.session_state:
    st.session_state.num_rooms = 5
if "city_name" not in st.session_state:
    st.session_state.city_name = "New York"
if 'use_case' not in st.session_state:
    st.session_state.use_case = None


with st.sidebar:
    st.header("Agent Configuration")
    
    # Industry selection
    industry = st.selectbox("Industry", [
    "",  # Empty/default
    "Energy Management in Buildings",
    "Public Sector"
], format_func=lambda x: "Select Industry" if x == "" else x)
    st.session_state.industry = industry
    
    # Use Case selection
    # Define use cases by industry
    industry_use_case_map = {
    "Energy Management in Buildings": [
        "Thermal Comfort Analysis", 
        "Energy Optimization", 
        "Space Optimization", 
        "Layout Planning"
    ],
    "Public Sector": [
        "Unemployment Policy", 
      "Healthcare Policy"
    ]
}

# Get relevant use cases based on selected industry
    use_case_options = industry_use_case_map.get(industry, [])
    use_case = st.selectbox("Use Case", [""] + use_case_options)
    st.session_state.use_case = use_case


    # Define agent tools, platforms, LLMs, frameworks and libraries for each use case
    agent_data = {
        "Thermal Comfort Analysis": {
            "tools": ["Thermal Comfort Calculator", "Weather API"],
            "platforms": ["CrewAI", "LangChain"],
            "llms": ["gpt-3.5-turbo", "gpt-4", "llama-2", "mistral-7b", "deepseek-coder"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["pythermalcomfort", "pandas", "requests"]
        },
        "Energy Optimization": {
            "tools": ["Energy Forecasting", "Weather API"],
            "platforms": ["CrewAI", "LangChain"],
            "llms": ["gpt-3.5-turbo", "gpt-4", "llama-2", "mistral-7b", "deepseek-coder"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["numpy", "scikit-learn", "pandas", "requests"]
        },
        "Space Optimization": {
            "tools": ["Space Analytics", "Occupancy Prediction"],
            "platforms": ["CrewAI", "LangChain"],
            "llms": ["gpt-3.5-turbo", "gpt-4", "llama-2", "mistral-7b", "deepseek-coder"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["pandas", "statsmodels", "scikit-learn"]
        },
        "Layout Planning": {
            "tools": ["Layout Recommendation"],
            "platforms": ["CrewAI", "langChain"],
            "llms": ["gpt-3.5-turbo", "gpt-4", "llama-2", "mistral-7b", "deepseek-coder"],
            "frameworks": ["CrewAI", "LangChain"],
            "libraries": ["streamlit", "numpy", "plotly"]
        },
        "Unemployment Policy": {
    "tools": [
        "Search and Scrape Job Postings",
        "Analyze Demographics",
        "Provide Career Guidance"
    ],
    "platforms": ["CrewAI", "langChain"],
    "llms": ["gpt-3.5-turbo", "gpt-4", "llama-2", "mistral-7b", "deepseek-coder"],
    "frameworks": ["CrewAI", "LangChain"],
    "libraries": ["pandas", "requests", "crewai_tools"]
}, 
      "Healthcare Policy": {
    "tools": ["Scrape Policy Websites", "Match Policy to Profile", "Recommend Care Action"],
    "platforms": ["CrewAI", "LangChain"],
    "llms": ["gpt-3.5-turbo", "gpt-4","llama-3.3-70b-versatile","mistral-saba-24b","deepseek-r1-distill-llama-70b"],
    "frameworks": ["CrewAI", "LangChain"],
    "libraries": ["pandas", "requests", "crewai_tools"]
}

    }

    # Get the selected use case data
    tool_ui_map = {
    "Thermal Comfort Analysis": [
        "Get Building Summary",
        "Get Weather Summary"
    ],
    "Energy Optimization": [
        "Simulate Energy Reduction",
        "Explain Energy Forecast"
    ],
    "Space Optimization": [
        "Predict Occupancy Trend",
        "Suggest Room Consolidation",
        "Suggest Space Rezoning"
    ],
    "Layout Planning": [
        "Recommend Layout Plan",
        "Identify Multiuse Zones",
        "Room Function Mapping"
    ],
    "Unemployment Policy": [
        "Search and Scarpe Job Posting",
        "Analyze Demographics",
        "Provide Career Guidance"
    ], 
      "Healthcare Policy": [
        "Scrape Policy Websites",
        "Match Policy to Profile",
        "Recommend Care Action"
    ]
}

    selected_tool_labels = tool_ui_map.get(use_case, [])
    default_tool = selected_tool_labels[0] if selected_tool_labels else None
    st.multiselect("Agent Tools", options=selected_tool_labels, default=[default_tool] if default_tool else [])


    # Fallback if use_case is not found
    agent_info = agent_data.get(use_case, {
    "tools": [],
    "platforms": [],
    "llms": [],
    "frameworks": [],
    "libraries": []
})

    agent_platform = st.selectbox("Agent Platform", agent_info["platforms"])
    agent_llm = st.selectbox("Agent LLM", agent_info["llms"])
    agent_framework = st.selectbox("Agent Framework", agent_info["frameworks"])
    agent_libraries = st.multiselect("Agent Libraries", agent_info["libraries"])


    agent_tools_table_data = {
    "Thermal Comfort Analysis": [
        {
            "Agent": "Thermal Comfort Analyst",
            "Tool": "Get Building Summary",
            "Function": "Provides average environmental and occupancy stats"
        },
        {
            "Agent": "Thermal Comfort Analyst",
            "Tool": "Get Weather Summary",
            "Function": "Fetches current weather data for given city coordinates"
        }
    ],
    "Energy Optimization": [
        {
            "Agent": "Energy Optimization Engineer",
            "Tool": "Simulate Energy Reduction",
            "Function": "Estimates energy savings from optimized HVAC & lighting"
        },
        {
            "Agent": "Energy Optimization Engineer",
            "Tool": "Explain Energy Forecast",
            "Function": "LLM-based forecast based on room and weather conditions"
        }
    ],
    "Space Optimization": [
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Get Building Summary",
            "Function": "Summarizes average metrics from building data"
        },
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Predict Occupancy Trend",
            "Function": "Forecasts room occupancy using ARIMA"
        },
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Suggest Room Consolidation",
            "Function": "Identifies underused rooms for merging/closing"
        },
        {
            "Agent": "Space Optimization Assistant",
            "Tool": "Suggest Space Rezoning",
            "Function": "Clusters rooms into functional zones using KMeans"
        }
    ],
    "Layout Planning": [
        {
            "Agent": "Layout Planner",
            "Tool": "Recommend Layout Plan",
            "Function": "Suggests layout changes to optimize space utilization"
        },
        {
            "Agent": "Layout Planner",
            "Tool": "Identify Multiuse Zones",
            "Function": "Detects rooms suitable for multi-functional use"
        },
        {
            "Agent": "Layout Planner",
            "Tool": "Room Function Mapping",
            "Function": "Assigns room roles based on occupancy, light, and energy"
        }
    ],
    "Unemployment Policy": [
    {
        "Agent": "Job Market Monitoring Agent",
        "Tool": "Search and Scrape Job Postings",
        "Function": "Extracts live job data from job boards"
    },
    {
        "Agent": "Demographic Impact Agent",
        "Tool": "Analyze Demographics",
        "Function": "Fetches unemployment rate by age, education, ethnicity"
    },
    {
        "Agent": "Citizen Guidance Agent",
        "Tool": "Provide Career Guidance",
        "Function": "Suggests training and job opportunities based on user profile"
    }
],
      "Healthcare Policy": [
    {
        "Agent": "Policy Update Agent",
        "Tool": "Scrape Policy Websites",
        "Function": "Scans CMS and state health sites for new policy updates"
    },
    {
        "Agent": "Eligibility Matching Agent",
        "Tool": "Match Policy to Profile",
        "Function": "Flags new eligibility based on user demographics"
    },
    {
        "Agent": "Personal Health Advisor Agent",
        "Tool": "Recommend Care Action",
        "Function": "Suggests next steps like clinic visits or how to apply"
    }
]

}

    # Default table when no use case is selected
    default_llm_config = pd.DataFrame({
    "Agent": [
        "Thermal Comfort Analyst",
        "Energy Optimization Engineer",
        "Space Optimization Assistant",
        "Layout Planner"
    ],
    "LLM Used": [
        "gpt-3.5-turbo, gpt-4",
        "gpt-4, claude-3-opus",
        "gpt-4-turbo, claude-3-sonnet",
        "gpt-3.5-turbo, gpt-4-turbo"
    ],
    "Purpose": [
        "Thermal analysis based on ISO/ASHRAE standards",
        "Forecasts and cost-saving strategy generation",
        "Occupancy forecasting, clustering explanation",
        "Layout suggestions and room function mapping"
    ]
})
    
    default_llm_config_public = pd.DataFrame({
    "Use Case": [
        "Unemployment Policy",
        "Healthcare Policy"
    ],
    "LLM Used": [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo"
    ],
    "Purpose": [
        "Job scraping, demographic trend analysis, career guidance generation",
        "Track policy updates, assess eligibility, and recommend care actions"
    ]
})


# Use-case-specific LLM mapping
    agent_llm_table_data = {
    "Thermal Comfort Analysis": [
        {
            "Agent": "Thermal Comfort Analyst",
            "LLM Used": "gpt-3.5-turbo, gpt-4",
            "Purpose": "Thermal analysis based on ISO/ASHRAE standards"
        }
    ],
    "Energy Optimization": [
        {
            "Agent": "Energy Optimization Engineer",
            "LLM Used": "gpt-4, claude-3-opus",
            "Purpose": "Forecasts and cost-saving strategy generation"
        }
    ],
    "Space Optimization": [
        {
            "Agent": "Space Optimization Assistant",
            "LLM Used": "gpt-4-turbo, claude-3-sonnet",
            "Purpose": "Occupancy forecasting, clustering explanation"
        }
    ],
    "Layout Planning": [
        {
            "Agent": "Layout Planner",
            "LLM Used": "gpt-3.5-turbo, gpt-4-turbo",
            "Purpose": "Layout suggestions and room function mapping"
        }
    ],
    "Unemployment Policy": [
    {
        "Agent": "Job Market Monitoring Agent",
        "LLM Used": "gpt-3.5-turbo",
        "Purpose": "Scrape and summarize job market insights"
    },
    {
        "Agent": "Demographic Impact Agent",
        "LLM Used": "gpt-3.5-turbo",
        "Purpose": "Analyze demographic trends in unemployment"
    },
    {
        "Agent": "Citizen Guidance Agent",
        "LLM Used": "gpt-3.5-turbo",
        "Purpose": "Offer personalized retraining/job advice"
    }
], 
      "Healthcare Policy": [
    {
        "Agent": "Policy Update Agent",
        "LLM Used": "gpt-3.5-turbo",
        "Purpose": "Track policy changes from health agencies"
    },
    {
        "Agent": "Eligibility Matching Agent",
        "LLM Used": "gpt-3.5-turbo",
        "Purpose": "Match updated policies to user profiles"
    },
    {
        "Agent": "Personal Health Advisor Agent",
        "LLM Used": "gpt-3.5-turbo",
        "Purpose": "Provide actionable care steps"
    }
      ]
}



    # Add the reset chat button
    st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    if st.button("Clear/Reset", key="reset_chat_button"):
        # Clear the messages in session state
        st.session_state.messages = []
        # Clear the context information
        st.session_state.available_data = []
        st.rerun()


    # --- LOGO CSS for bottom placement and round/white style ---
    st.markdown("""
    <style>
    /* Make sidebar a flex column */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    /* Spacer grows to push logos down */
    .spacer {
        flex: 1 1 auto;
        height: 8px;
    }
    .sidebar-bottom-logos {
        width: 100%;
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 24px;
    }
    .sidebar-bottom-logos img {
        width: 45px;
        height: 45px;
        border-radius: 10%;
        background: #fff;
        padding: 0px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        object-fit: scale-down;
        border: 2px solid #fff;
    }
    [data-testid="stButton"] button {
    background-color: #33cc33;
    color: white;
    font-weight: bold;
    width: 100%;
    margin-bottom: 15px;
}
    </style>
    """, unsafe_allow_html=True)

    # --- SPACER to push logos to bottom ---
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # --- LOGOS PLACEMENT ---
    st.markdown('''
<div class="sidebar-bottom-logos">
    <img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" alt="AWS Logo">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/googlecloud/googlecloud-original.svg" alt="GCP Logo">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg" alt="Azure Logo">
</div>
''', unsafe_allow_html=True)




@st.cache_resource
def get_openai_client(api_key):
    if api_key and api_key.startswith('sk-'):
        return OpenAI(api_key=api_key)
    return None

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['current']
    return None

@st.cache_data
def get_sample_data():
    sample_data = {
        "current_temperature": [23.5],
        "humidity_level": [45.0],
        "ambient_light_level": [500],
        "current_power_consumption": [45.5],
        "energy_tariff_rate": [0.15],
        "hvac_status": ["On - Cooling"],
        "lighting_status": ["Dimmed"],
        "appliance_status": ["Essential Only"],
        "lighting_power_usage": [12.3],
        "appliance_power_usage": [18.7]
    }
    return pd.DataFrame(sample_data)

def calculate_thermal_metrics(inputs):
    pmv_result = pmv_ppd_iso(tdb=inputs['tdb'], tr=inputs['tr'], vr=inputs['vr'],
                           rh=inputs['rh'], met=inputs['met'], clo=inputs['clo'])
    tdb_for_utci = inputs.get('outdoor_temp', inputs['tdb'])
    utci_result = utci(tdb=tdb_for_utci, tr=inputs['tr'], v=inputs['vr'], rh=inputs['rh'])

    return {
        'pmv': round(pmv_result.pmv, 2),
        'ppd': round(pmv_result.ppd, 1),
        'utci': round(utci_result.utci, 1),
        'utci_category': utci_result.stress_category if hasattr(utci_result, "stress_category") else "Unknown"
    }

def _get_building_summary_internal() -> str:
    try:
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
    except Exception as e:
        # Fallback if file doesn't exist
        if st.session_state.rooms_df is not None:
            df = st.session_state.rooms_df
            return (
                f"Average Temperature: {df['Temperature'].mean():.2f} °C\n"
                f"Average CO₂ Level: {df['CO2'].mean():.2f} ppm\n"
                f"Average Light Level: {df['Light'].mean():.2f} Lux\n"
                f"Average Humidity: {df['Humidity'].mean():.2f} %\n"
                f"Average Occupancy: {df['Occupancy'].mean():.2f}"
            )
        return "Building summary data not available."

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

def call_openai_fallback(prompt: str, api_key: str, model: str, label: str) -> str:
    try:
        client = OpenAI(api_key=api_key)
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
        return f"[{label}] OpenAI fallback failed: {e}"

def process_chat_query(query):
    if not st.session_state.openai_api_key:
        return "Please enter a valid OpenAI API key to use the chat feature."
    
    # Build context from available data and reports
    context = "You are an expert in building optimization. Here is the relevant context:\n\n"
    
    # Add thermal data if available
    if st.session_state.thermal_data:
        thermal_data = st.session_state.thermal_data
        context += f"THERMAL DATA:\n"
        context += f"Building Type: {thermal_data['building_type']}\n"
        context += f"Season: {thermal_data['season']}\n"
        context += f"Indoor Temperature: {thermal_data['tdb']}°C\n"
        context += f"Relative Humidity: {thermal_data['rh']}%\n"
        context += f"PMV: {thermal_data['pmv']}\n"
        context += f"PPD: {thermal_data['ppd']}%\n"
        context += f"UTCI: {thermal_data['utci']}°C\n"
        context += f"UTCI Category: {thermal_data['utci_category']}\n\n"
    
    # Add generated reports if available
    if "thermal_analysis" in st.session_state.results:
        context += f"THERMAL ANALYSIS REPORT:\n{st.session_state.results['thermal_analysis']}\n\n"
    
    if "energy_optimization" in st.session_state.results:
        context += f"ENERGY OPTIMIZATION REPORT:\n{st.session_state.results['energy_optimization']}\n\n"
    
    if "space_optimization" in st.session_state.results:
        context += f"SPACE OPTIMIZATION REPORT:\n{st.session_state.results['space_optimization']}\n\n"
    
    if "layout_recommendation" in st.session_state.results:
        context += f"LAYOUT RECOMMENDATIONS:\n{st.session_state.results['layout_recommendation']}\n\n"
    
    # Add building summary if available
    try:
        building_summary = _get_building_summary_internal()
        context += f"BUILDING SUMMARY:\n{building_summary}\n\n"
    except:
        pass
    
    # Add weather data if available
    try:
        weather_summary = _get_weather_summary_internal(st.session_state.city_name)
        context += f"WEATHER SUMMARY:\n{weather_summary}\n\n"
    except:
        pass
    
    context += "Answer the user's question based on this information."
    
    try:
        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"



def create_thermal_agent():
    selected_labels = [
        "Get Building Summary",
        "Get Weather Summary"
    ]
    return Agent(
        role="Thermal Comfort Analyst",
        goal="Analyze indoor environmental parameters and generate technical thermal comfort reports",
        backstory="Expert in building science and ISO/ASHRAE thermal comfort standards",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )


def create_energy_agent():
    selected_labels = [
        "Simulate Energy Reduction",
        "Explain Energy Forecast"
    ]
    return Agent(
        role="Energy Optimization Engineer",
        goal="Recommend energy-saving actions while maintaining thermal comfort",
        backstory="Specialist in building energy systems and cost-benefit analysis",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )


def calculate_hvac_energy(inputs):
    """Calculate HVAC energy consumption based on thermal load"""
    temp_diff = abs(inputs['indoor_temp'] - inputs['setpoint_temp'])

    # Cooling energy (simplified model)
    if inputs['indoor_temp'] > inputs['setpoint_temp']:
        cooling_load = temp_diff * inputs['building_area'] * 0.05  # kW
        cop_cooling = 3.5  # Coefficient of Performance
        cooling_energy = cooling_load / cop_cooling
    else:
        cooling_energy = 0

    # Heating energy
    if inputs['indoor_temp'] < inputs['setpoint_temp']:
        heating_load = temp_diff * inputs['building_area'] * 0.06  # kW
        efficiency = 0.85  # Heating efficiency
        heating_energy = heating_load / efficiency
    else:
        heating_energy = 0

    return {
        'cooling_energy_kw': round(cooling_energy, 2),
        'heating_energy_kw': round(heating_energy, 2),
        'total_hvac_energy_kw': round(cooling_energy + heating_energy, 2)
    }

def calculate_lighting_energy(inputs):
    """Calculate lighting energy based on occupancy and daylight"""
    base_lighting_power = inputs['lighting_power_density']  # W/m²
    area = inputs['building_area']
    occupancy_factor = inputs['occupancy_rate']
    daylight_factor = max(0.2, 1 - (inputs['daylight_lux'] / 1000))

    lighting_energy = (base_lighting_power * area * occupancy_factor *
                      daylight_factor) / 1000  # Convert to kW

    return {
        'lighting_energy_kw': round(lighting_energy, 2),
        'daylight_savings_percent': round((1 - daylight_factor) * 100, 1)
    }

def calculate_energy_savings(current_energy, optimized_settings):
    """Calculate potential energy savings from optimization"""
    # Temperature setpoint optimization
    temp_savings = 0
    if optimized_settings['temp_adjustment'] != 0:
        temp_savings = abs(optimized_settings['temp_adjustment']) * 0.08 * current_energy['hvac']

    # Lighting optimization
    lighting_savings = 0
    if optimized_settings['lighting_reduction'] > 0:
        lighting_savings = current_energy['lighting'] * optimized_settings['lighting_reduction']

    # Equipment optimization
    equipment_savings = 0
    if optimized_settings['equipment_efficiency'] > 1:
        equipment_savings = current_energy['equipment'] * (1 - 1/optimized_settings['equipment_efficiency'])

    total_savings = temp_savings + lighting_savings + equipment_savings

    return {
        'hvac_savings_kw': round(temp_savings, 2),
        'lighting_savings_kw': round(lighting_savings, 2),
        'equipment_savings_kw': round(equipment_savings, 2),
        'total_savings_kw': round(total_savings, 2),
        'savings_percentage': round((total_savings / sum(current_energy.values())) * 100, 1)
    }

def extract_energy_findings(inputs: dict, metrics: dict) -> list:
    findings = []

    hvac_kw = metrics['hvac_energy_kw']
    total_kw = metrics['total_current_energy_kw']
    hvac_ratio = hvac_kw / total_kw if total_kw > 0 else 0

    if hvac_ratio > 0.5:
        findings.append(
            f"HVAC consumes {hvac_ratio*100:.1f}% of total energy. Temperature setpoint is {inputs['setpoint_temp']}°C and indoor temperature is {inputs['current_temperature']}°C."
        )

    if inputs['ambient_light_level'] > 500 and inputs['lighting_power_usage'] > 8:
        findings.append(
            f"Ambient light is {inputs['ambient_light_level']} lux, but lighting power is {inputs['lighting_power_usage']} kW — possibly excessive daylight usage."
        )

    if inputs['appliance_power_usage'] > 15:
        findings.append(
            f"Appliance load is high at {inputs['appliance_power_usage']} kW. May indicate inefficient or continuous appliance operation."
        )

    if inputs['occupancy_rate'] < 0.3 and total_kw > 30:
        findings.append(
            f"Occupancy rate is low ({inputs['occupancy_rate']*100:.0f}%), but load is high ({total_kw:.1f} kW) — suggesting overuse in underutilized spaces."
        )

    return findings


def calculate_comprehensive_energy_metrics(inputs):
    """Calculate all energy metrics without LLM"""

    # HVAC calculations
    hvac_metrics = calculate_hvac_energy({
        'indoor_temp': inputs['current_temperature'],
        'setpoint_temp': inputs.get('setpoint_temp', 22),
        'building_area': inputs.get('building_area', 100)
    })

    # Lighting calculations
    lighting_metrics = calculate_lighting_energy({
        'lighting_power_density': 10,  # W/m²
        'building_area': inputs.get('building_area', 100),
        'occupancy_rate': inputs.get('occupancy_rate', 0.7),
        'daylight_lux': inputs['ambient_light_level']
    })

    # Current energy breakdown
    current_energy = {
        'hvac': inputs['current_power_consumption'] * 0.6,
        'lighting': inputs['lighting_power_usage'],
        'equipment': inputs['appliance_power_usage']
    }

    # Optimization scenarios
    optimized_settings = {
        'temp_adjustment': 2 if inputs['current_temperature'] > 24 else -2,
        'lighting_reduction': 0.3 if inputs['ambient_light_level'] > 300 else 0.1,
        'equipment_efficiency': 1.2
    }

    savings_metrics = calculate_energy_savings(current_energy, optimized_settings)

    # Cost calculations
    energy_rate = inputs['energy_tariff_rate']
    annual_cost_current = sum(current_energy.values()) * 24 * 365 * energy_rate
    annual_savings = savings_metrics['total_savings_kw'] * 24 * 365 * energy_rate

    # Merge all metrics into one object
    metrics = {
        'hvac_energy_kw': hvac_metrics['total_hvac_energy_kw'],
        'lighting_energy_kw': lighting_metrics['lighting_energy_kw'],
        'total_current_energy_kw': sum(current_energy.values()),
        'potential_savings_kw': savings_metrics['total_savings_kw'],
        'savings_percentage': savings_metrics['savings_percentage'],
        'annual_cost_current': round(annual_cost_current, 2),
        'annual_savings': round(annual_savings, 2),
        'payback_period_years': 2.5  # Typical for energy efficiency measures
    }

    # Add extracted observations
    metrics['recommendations'] = extract_energy_findings(inputs, metrics)

    return metrics




def create_space_optimization_agent():
    return Agent(
        role="Space Optimization Assistant",
        goal="Generate a detailed optimization report from tool outputs",
        backstory="Expert in spatial efficiency and smart zoning.",
        tools=[],  
        verbose=True,
        allow_delegation=False
    )



def create_layout_recommendation_agent():
    selected_labels = [
        "Recommend Layout Plan",
        "Identify Multiuse Zones",
        "Room Function Mapping"
    ]
    return Agent(
        role="Layout Planner",
        goal="Analyze room data and suggest long-term layout changes for flexibility and efficiency",
        backstory="Specialist in spatial design and dynamic reconfiguration based on usage metrics",
        tools=[tool_function_map[label] for label in selected_labels],
        verbose=True,
        allow_delegation=False
    )

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()



def create_thermal_analysis_task(inputs, agent):
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    metrics = calculate_thermal_metrics(inputs)

    pmv_val = metrics['pmv']
    ppd = metrics['ppd']
    utci = metrics['utci']
    utci_category = metrics['utci_category'].lower()

    outdoor_temp = inputs.get('outdoor_temp', inputs['tdb'])

    recommended_temp = get_recommended_temperature_with_outdoor(
        current_temp=inputs['tdb'],
        tr=inputs['tr'],
        vr=inputs['vr'],
        rh=inputs['rh'],
        met=inputs['met'],
        clo=inputs['clo'],
        outdoor_temp=outdoor_temp
    )

    t_comf, t_min, t_max = get_adaptive_comfort_temp(outdoor_temp)

    context = """
You are a thermal comfort analysis expert. You are provided with scientific indoor environment parameters for a room.
DO NOT mention unrelated metrics like CO₂ levels, light intensity, cloud cover, or any environmental data not shown in the prompt.
Base your evaluation strictly on: temperature, humidity, air velocity, metabolic rate, clothing insulation, and PMV/PPD/UTCI values.
On the first mention of PMV, PPD, UTCI, MET, or CLO, always include the full form in parentheses.
Abbreviation Definitions:
- PMV: Predicted Mean Vote
- PPD: Predicted Percentage of Dissatisfied
- UTCI: Universal Thermal Climate Index
- MET: Metabolic Rate
- CLO: Clothing Insulation
Provide actionable and technically sound advice tailored only to the data shown.
Reference ASHRAE 55 adaptive model logic and thermal science where appropriate.
"""

    query = f"""
Room Type: {inputs.get('building_type', 'Unknown')}
Season: {inputs.get('season', 'Unknown')}
Number of Occupants: {inputs.get('num_people', 1)}

Indoor Conditions:
- Air Temp: {inputs['tdb']}°C
- Radiant Temp: {inputs['tr']}°C
- Relative Humidity: {inputs['rh']}%
- Air Velocity: {inputs['vr']} m/s
- Metabolic Rate: {inputs['met']} met
- Clothing Level: {inputs['clo']} clo
- Outdoor Temp: {outdoor_temp}°C

Thermal Comfort Metrics:
- Predicted Mean Vote: {pmv_val}, Predicted Percentage of Dissatisfied: {ppd}%
- Universal Thermal Climate Index: {utci}°C → {utci_category}
- Predicted Mean Vote-based Recommended Temp: {recommended_temp}°C
- ASHRAE 55 Adaptive Range: {t_min}°C – {t_max}°C
- Adaptive Comfort Temp (t_comf): {t_comf}°C
Instructions:
1. Provide an in-depth comfort analysis explaining how each metric (temperature, humidity, air speed, clothing, metabolic rate) affects the comfort level.
2. If any parameters seem contradictory (e.g., comfortable PMV but high UTCI), explain the contradiction clearly.
3. Evaluate the comfort level based on the given values only.
4. Identify specific reasons for discomfort and perform a root-cause diagnosis if PPD > 10%.
5. Provide actionable expert-level suggestions prioritized as:
   - Immediate (e.g., HVAC tuning, localized changes)
   - Medium-term (e.g., behavioral changes, occupancy scheduling)
   - Long-term (e.g., structural improvements, insulation)
6. Include how different occupant activity levels (sedentary vs active) might perceive this environment.
7. **IMPORTANT**: On the first mention of each of the following abbreviations, write their full form in parentheses:
   - PMV → Predicted Mean Vote
   - PPD → Predicted Percentage of Dissatisfied
   - UTCI → Universal Thermal Climate Index
   - MET → Metabolic Rate
   - CLO → Clothing Insulation
   For example: "PMV (Predicted Mean Vote) of 0.85 indicates..."
8. After the first use, you may refer to these abbreviations without repeating the full form.
9. Do not include any environmental data not shown here.


"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ],
        temperature=0.4,
        max_tokens=1200
    )
    raw_output = response.choices[0].message.content.strip()
    final_output = enforce_full_forms(raw_output)



    return Task(
        description= final_output,
        agent=agent,
        expected_output="Detailed thermal comfort evaluation with scientifically backed suggestions."
    )


def create_combined_thermal_analysis_task(buildings_df, agent):
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_building_descriptions = []
    num_comfortable = 0
    num_uncomfortable = 0
    air_movement_issues = 0
    humidity_issues = 0
    temp_out_of_range = 0

    for idx, row in buildings_df.iterrows():
        inputs = {
            'tdb': row['air_temperature_celsius'],
            'tr': row['mean_radiant_temperature_celsius'],
            'vr': row['air_velocity_meters_per_second'],
            'rh': row['relative_humidity_percent'],
            'met': row['metabolic_rate_met'],
            'clo': row['clothing_insulation_clo'],
            'outdoor_temp': row['outdoor_temperature_celsius']
        }

        metrics = calculate_thermal_metrics(inputs)

        recommended_temp = get_recommended_temperature_with_outdoor(
            current_temp=inputs['tdb'],
            tr=inputs['tr'],
            vr=inputs['vr'],
            rh=inputs['rh'],
            met=inputs['met'],
            clo=inputs['clo'],
            outdoor_temp=inputs['outdoor_temp']
        )

        t_comf, t_min, t_max = get_adaptive_comfort_temp(inputs['outdoor_temp'])

        if metrics['ppd'] <= 10:
            num_comfortable += 1
        else:
            num_uncomfortable += 1

        if inputs['vr'] < 0.2:
            air_movement_issues += 1

        if inputs['rh'] < 40 or inputs['rh'] > 60:
            humidity_issues += 1

        if not (t_min <= recommended_temp <= t_max):
            temp_out_of_range += 1

        # Format the LLM prompt to match single-building tone
        context = """
You are a thermal comfort analysis expert. You are provided with scientific indoor environment parameters for a room.
DO NOT mention unrelated metrics like CO₂ levels, light intensity, cloud cover, or any environmental data not shown in the prompt.
Base your evaluation strictly on: temperature, humidity, air velocity, metabolic rate, clothing insulation, and PMV/PPD/UTCI values.
On the first mention of PMV, PPD, UTCI, MET, or CLO, always include the full form in parentheses.
Abbreviation Definitions:
- PMV: Predicted Mean Vote
- PPD: Predicted Percentage of Dissatisfied
- UTCI: Universal Thermal Climate Index
- MET: Metabolic Rate
- CLO: Clothing Insulation
Provide actionable and technically sound advice tailored only to the data shown.
Reference ASHRAE 55 adaptive model logic and thermal science where appropriate.
"""

        query = f"""
Room Type: {row.get('building_type', 'Unknown')}
Season: {row.get('season', 'Unknown')}
Number of Occupants: {row.get('number_of_occupants', 1)}

Indoor Conditions:
- Air Temp: {inputs['tdb']}°C
- Radiant Temp: {inputs['tr']}°C
- Relative Humidity: {inputs['rh']}%
- Air Velocity: {inputs['vr']} m/s
- Metabolic Rate (MET): {inputs['met']} met
- Clothing Insulation (CLO): {inputs['clo']} clo
- Outdoor Temp: {inputs['outdoor_temp']}°C

Thermal Comfort Metrics:
- PMV (Predicted Mean Vote): {metrics['pmv']}
- PPD (Predicted Percentage of Dissatisfied): {metrics['ppd']}%
- UTCI (Universal Thermal Climate Index): {metrics['utci']}°C → {metrics['utci_category']}
- Predicted Mean Vote-based Recommended Temp: {recommended_temp}°C
- ASHRAE 55 Adaptive Range: {t_min}°C – {t_max}°C
- Adaptive Comfort Temp (t_comf): {t_comf}°C

Instructions:
1. Provide an in-depth comfort analysis explaining how each metric (temperature, humidity, air speed, clothing, metabolic rate) affects the comfort level.
2. If any parameters seem contradictory (e.g., comfortable PMV but high UTCI), explain the contradiction clearly.
3. Evaluate the comfort level based on the given values only.
4. Identify specific reasons for discomfort and perform a root-cause diagnosis if PPD > 10%.
5. Provide actionable expert-level suggestions prioritized as:
   - Immediate (e.g., HVAC tuning, localized changes)
   - Medium-term (e.g., behavioral changes, occupancy scheduling)
   - Long-term (e.g., structural improvements, insulation)
6. Include how different occupant activity levels (sedentary vs active) might perceive this environment.
7. **IMPORTANT**: On the first mention of each of the following abbreviations, write their full form in parentheses:
   - PMV → Predicted Mean Vote
   - PPD → Predicted Percentage of Dissatisfied
   - UTCI → Universal Thermal Climate Index
   - MET → Metabolic Rate
   - CLO → Clothing Insulation
   For example: "PMV (Predicted Mean Vote) of 0.85 indicates..."
8. After the first use, you may refer to these abbreviations without repeating the full form.
9. Do not include any environmental data not shown here.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": query}
                ],
                temperature=0.4,
                max_tokens=1200
            )
            raw_response = response.choices[0].message.content.strip()
            building_report = enforce_full_forms(raw_response)

        except Exception as e:
            building_report = f"⚠️ Error analyzing building {row.get('building_id', idx)}: {str(e)}"

        all_building_descriptions.append(f"\n--- Building {idx+1} Analysis ---\n{building_report}")

    # General summary in same tone
    general_summary_prompt = f"""
You are a building comfort optimization expert. Based on the following:
- Comfortable buildings: {num_comfortable}
- Uncomfortable buildings: {num_uncomfortable}
- Issues observed:
  - Low air movement: {air_movement_issues} cases
  - Humidity outside 40–60%: {humidity_issues} cases
  - Temp outside ASHRAE 55 comfort range: {temp_out_of_range} cases

Provide a scientific, short summary and universal advice to improve thermal comfort in this building group. Maintain the same tone as individual building reports.
"""

    general_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You summarize comfort trends in multiple buildings."},
            {"role": "user", "content": general_summary_prompt}
        ],
        temperature=0.4,
        max_tokens=1200
    )

    summary_report = enforce_full_forms(general_response.choices[0].message.content.strip())


    final_description = f"""
📘 General Comfort Summary:
{summary_report}

📊 Detailed Comfort Analysis per Building:
{''.join(all_building_descriptions)}
""".strip()

    return Task(
        description=final_description,
        agent=agent,
        expected_output="Detailed thermal comfort evaluation with scientifically backed suggestions."
    )





def run_multi_building_thermal_analysis(buildings_df):
    agent = create_thermal_agent()
    task = create_combined_thermal_analysis_task(buildings_df, agent)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential
    )
    return crew.kickoff(), buildings_df

if "thermal_analysis" not in st.session_state:
    st.session_state.thermal_analysis = ""


def fix_malformed_units(text):
    # Fix broken monetary expressions (e.g., '0.30 per kWh' → '$0.30/kWh')
    text = re.sub(r"(?<!\$)(\d+\.\d+)\s*per\s*kWh", r"$\1/kWh", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\$)(\d+\.\d+)\s*/\s*hour", r"$\1/hour", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\$)(\d+\.\d+)\s*/\s*day", r"$\1/day", text, flags=re.IGNORECASE)

    # Compact broken unit words
    text = re.sub(r"k\s*W\s*h", "kWh", text, flags=re.IGNORECASE)
    text = re.sub(r"k\s*W", "kW", text, flags=re.IGNORECASE)
    text = re.sub(r"\$ ?(\d+(?:,\d{3})*(?:\.\d+)?)", r"$\1", text)  # e.g. $ 1,200.50 → $1200.50

    # Clean markdown math glitches (like split stars or units)
    text = re.sub(r"\*\*\s*\$", "**$", text)  # Merge split bold and $
    text = re.sub(r"\$\s*(\d+)", r"$\1", text)  # Remove space between $ and number

    # Remove newlines between numbers and units like:
    # "40.92\n/\nday" → "40.92/day"
    text = re.sub(r"(\d+\.\d+)\s*[\r\n]+/\s*[\r\n]*(day|hour)", r"\1/\2", text, flags=re.IGNORECASE)

    # Collapse multiple spaces between numbers and units
    text = re.sub(r"(\d+\.\d+)\s+(kWh|kW|/hour|/day)", r"\1\2", text)

    return text

def create_energy_optimization_task(energy_inputs, agent, pricing_scheme):
    # Initialize OpenAI client
    client = OpenAI(api_key=st.session_state.get("openai_api_key"))

    # === Step 1: Forecast Energy Usage ===
    hourly_series = st.session_state.get("hourly_energy_series", None)
    if hourly_series is None or len(hourly_series) < 24:
        raise ValueError("Need at least 24 hours of historical hourly energy usage for forecasting.")

    forecast_df = forecast_energy_usage_arima(hourly_series, steps=6, plot=False)

    # === Step 2: Predict Optimized Energy Reduction ===
    reduction_result = predict_energy_reduction_regression(energy_inputs)

    # === Step 3: Dynamic Pricing Estimate ===
    pricing_result = calculate_dynamic_pricing_cost(hourly_series[-24:], pricing_scheme)

    # === Step 4: Compute Comprehensive Energy Metrics ===
    metrics = calculate_comprehensive_energy_metrics(energy_inputs)

    # === Step 5: Build Prompt ===
    context = """
You are a senior energy optimization expert helping a building manager understand energy reports.

You are provided real data from forecasting, system breakdowns, and cost modeling.
Your job is not to repeat numbers — instead, use them to draw **insights**.

🎯 Your report must include:
1. Forecast interpretation: explain the **implications** of the 6-hour trend. What does it suggest about building behavior, operations, or occupancy?
2. Optimization analysis: Compare current and optimized usage. Identify which component (HVAC, lighting, appliances) drives overuse — and WHY it might be overused (e.g., high delta between indoor and setpoint temp, lighting running during daylight, appliances running at low occupancy).
3. Recommendations: Provide **concrete, data-driven** actions. For each action, briefly explain how it impacts efficiency and when to implement it.
4. Pricing response: Use the actual cost, tariff spread, and usage pattern to suggest a load shifting strategy. Quantify it in savings per hour/day.

🧠 Use only the numbers given.
💬 Speak like a human expert — avoid robotic summaries.
📌 Expand beyond data — explain patterns, anomalies, or improvement paths.
🧾 Format instructions:
- Always format currency savings as **$X/hour** and **$Y/day**, not as slashed formats like `3/hour` or spread across lines.
- Keep units like "kW", "$", and "%" on the same line as the number.
- Use Markdown-friendly symbols (like `**bold**`, `- bullet points`, and line breaks) to structure content clearly.

"""


    query = f"""
🔢 DATA SUMMARY:

📈 Forecast (Next 6 Hours):
{forecast_df.to_string(index=True, header=True)}

🔧 Optimization Prediction:
- Current: {reduction_result['current_power_kw']} kW
- Optimized: {reduction_result['predicted_optimized_power_kw']} kW
- Estimated Reduction: {reduction_result['estimated_reduction_kw']} kW ({reduction_result['reduction_percent']}%)

⚙ Load Breakdown:
- HVAC Load: {metrics['hvac_energy_kw']} kW
- Lighting Load: {metrics['lighting_energy_kw']} kW
- Total Load: {metrics['total_current_energy_kw']} kW

💵 Pricing Summary:
- Total Cost (Last 24h): ${pricing_result['total_cost']:.2f}
- High-Cost Usage:
{chr(10).join([f"{ts.strftime('%Y-%m-%d %H:%M')} — {usage:.1f} kWh @ ${rate:.2f}" for ts, usage, rate in pricing_result['high_cost_hours']]) or "None"}

🎯 Based on the above, please generate an in-depth energy report following the structure:
1. 📊 Forecast Summary
2. ⚙ Optimization Analysis
3. 🛠 Recommended Adjustments
4. 💸 Pricing Optimization

Use detailed reasoning. Do not repeat values without explaining them.
"""


    # === Step 6: Call LLM ===
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context.strip()},
            {"role": "user", "content": query.strip()}
        ],
        temperature=0.4,
        max_tokens=1600
    )

    raw_output = response.choices[0].message.content.strip()
    cleaned_output = fix_malformed_units(raw_output)
    task = Task(
        description=cleaned_output,
        agent=agent,
        expected_output="Structured optimization plan with forecast, adjustments, and pricing strategy."
    )
    return {
        "task": task,
        "forecast_df": forecast_df,
        "reduction_result": reduction_result,
    }


def create_multi_building_optimization_tasks(building_data_list, agent, pricing_scheme):
    client = OpenAI(api_key=st.session_state.get("openai_api_key"))
    all_tasks = []

    for idx, energy_inputs in enumerate(building_data_list):
        try:
            hourly_series = generate_synthetic_energy_series(energy_inputs['current_power_consumption'])

            if hourly_series is None or len(hourly_series) < 24:
                raise ValueError("Need at least 24 hours of historical hourly energy usage for forecasting.")

            forecast_df = forecast_energy_usage_arima(hourly_series, steps=6, plot=False)
            reduction_result = predict_energy_reduction_regression(energy_inputs)
            pricing_result = calculate_dynamic_pricing_cost(hourly_series[-24:], pricing_scheme)
            metrics = calculate_comprehensive_energy_metrics(energy_inputs)

            context = """
You are a senior energy optimization expert helping a building manager understand energy reports.

You are provided real data from forecasting, system breakdowns, and cost modeling.
Your job is not to repeat numbers — instead, use them to draw **insights**.

🎯 Your report must include:
1. Forecast interpretation: explain the **implications** of the 6-hour trend. What does it suggest about building behavior, operations, or occupancy?
2. Optimization Analysis:
   - Explicitly discuss the difference between current and optimized usage in **kW** and **%**.
   - Explain if the reduction is significant, small, or zero — and **why**.
   - If no reduction is predicted, state possible reasons (e.g., system already efficient, low baseline usage).
3. Recommendations: Provide **concrete, data-driven** actions. For each action, briefly explain how it impacts efficiency and when to implement it.
4. Pricing response: Use the actual cost, tariff spread, and usage pattern to suggest a load shifting strategy. Quantify it in savings per hour/day.

🧠 Use only the numbers given.
💬 Speak like a human expert — avoid robotic summaries.
📌 Expand beyond data — explain patterns, anomalies, or improvement paths.
🧾 Format instructions:
- Always format currency savings as **$X/hour** and **$Y/day**, not as slashed formats like `3/hour` or spread across lines.
- Keep units like "kW", "$", and "%" on the same line as the number.
- Use Markdown-friendly symbols (like `**bold**`, `- bullet points`, and line breaks) to structure content clearly.
"""

            query = f"""
🔢 DATA SUMMARY:

📈 Forecast (Next 6 Hours):
{forecast_df.to_string(index=True, header=True)}

🔧 Optimization Prediction:
- Current: {reduction_result['current_power_kw']} kW
- Optimized: {reduction_result['predicted_optimized_power_kw']} kW
- Estimated Reduction: {reduction_result['estimated_reduction_kw']} kW ({reduction_result['reduction_percent']}%)

⚙ Load Breakdown:
- HVAC Load: {metrics['hvac_energy_kw']} kW
- Lighting Load: {metrics['lighting_energy_kw']} kW
- Total Load: {metrics['total_current_energy_kw']} kW

💵 Pricing Summary:
- Total Cost (Last 24h): ${pricing_result['total_cost']:.2f}
- High-Cost Usage:
{chr(10).join([f"{ts.strftime('%Y-%m-%d %H:%M')} — {usage:.1f} kWh @ ${rate:.2f}" for ts, usage, rate in pricing_result['high_cost_hours']]) or "None"}

🎯 Based on the above, please generate an in-depth energy report following the structure:
1. 📊 Forecast Summary
2. ⚙ Optimization Analysis
3. 🛠 Recommended Adjustments
4. 💸 Pricing Optimization

Use detailed reasoning. Do not repeat values without explaining them.
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context.strip()},
                    {"role": "user", "content": query.strip()}
                ],
                temperature=0.4,
                max_tokens=1600
            )

            raw_output = response.choices[0].message.content.strip()
            cleaned_output = fix_malformed_units(raw_output)

            task = Task(
                description=cleaned_output,
                agent=agent,
                expected_output="Structured optimization plan with forecast, adjustments, and pricing strategy."
            )

            all_tasks.append({
                "building_index": idx + 1,
                "task": task,
                "forecast_df": forecast_df,
                "reduction_result": reduction_result
            })

        except Exception as e:
            all_tasks.append({
                "building_index": idx + 1,
                "task": None,
                "error": str(e)
            })

    return all_tasks

def get_recommended_temperature_with_outdoor(current_temp, tr, vr, rh, met, clo, outdoor_temp):
    from scipy.optimize import minimize_scalar

    def objective(temp):
        result = pmv_ppd_iso(tdb=temp, tr=tr, vr=vr, rh=rh, met=met, clo=clo)
        return abs(result.pmv)

    t_comf = 0.31 * outdoor_temp + 17.8
    t_min = round(t_comf - 2.5, 1)
    t_max = round(t_comf + 2.5, 1)

    # Fix invalid bound error
    lower = min(t_min, t_max)
    upper = max(t_min, t_max)
    if lower == upper:
        upper += 0.5  # Ensure there is a nonzero optimization range

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    return round(result.x, 1)


def get_adaptive_comfort_temp(outdoor_temp):
    """
    ASHRAE-55 Adaptive Comfort Model: for naturally ventilated buildings
    """
    t_comf = 0.31 * outdoor_temp + 17.8
    return round(t_comf, 1), round(t_comf - 2.5, 1), round(t_comf + 2.5, 1)


def create_space_optimization_task(rooms_df, agent):
    import os
    import json
    from openai import OpenAI
    import streamlit as st

    # ✅ Ensure the room data is available to all tools
    st.session_state["rooms_df"] = rooms_df

    # ✅ Run your tools (they rely on st.session_state["rooms_df"])
    occupancy_forecast = predict_occupancy_trend()
    consolidation_output = suggest_room_consolidation()
    rezoning_output = suggest_space_rezoning()

    # ✅ Format the raw room data as JSON string for LLM context
    formatted_data = json.dumps(rooms_df.to_dict(orient="records"), indent=2)

    # ✅ Set up the LLM client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ✅ System role and LLM prompt
    context = """
You are a senior space optimization analyst. You have been provided:
- Raw room data with environmental and occupancy metrics
- Forecasted occupancy trends
- Consolidation suggestions
- Room zoning recommendations

You must now synthesize this into a professional and actionable optimization plan.
"""

    query = f"""
📋 Raw Room Data (current snapshot):
{formatted_data}

📈 Forecasted Occupancy Trends (6-hour outlook):
{occupancy_forecast}

🔄 Room Consolidation Suggestions:
{consolidation_output}

📍 Space Re-zoning Summary:
{rezoning_output}

Instructions:
- Always reference the full 6-hour forecasted trend per room
- Mention the time frame (e.g., next 6 hours)
- Avoid vague language like "slight increase" — provide actual average and values
- Use the forecasted occupancy values exactly as given for each room.
- Use the current occupancy values from the dataset to compare with forecasts.
- Identify rooms at risk of underutilization (e.g., forecasted avg < 0.3 OR current very low).
- Recommend consolidation only for rooms showing both low forecast and low current usage.
- Justify zoning choices using room attributes like occupancy, humidity, light, power.

Respond with:
1. 📈 Occupancy Forecast Summary (mention specific rooms, trends, values)
2. 🔄 Room Consolidation Plan with reasoning
3. 📍 Space Rezoning Strategy — why rooms belong in each zone
4. 💡 Additional Expert Suggestions — HVAC, lighting, scheduling improvements
5. 📌 Final Action Plan: clear per-room next steps
"""


    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ],
        temperature=0.4,
        max_tokens=1600
    )

    return Task(
        description=response.choices[0].message.content.strip(),
        agent=agent,
        expected_output="Detailed optimization plan with room-specific actions based on tool outputs."
    )


def create_layout_recommendation_task(rooms_data, agent):
    return Task(
        description=f"""Analyze the following room data and provide layout recommendations:
        
        {rooms_data}
        
        Focus on:
        1. Identifying optimal room functions based on usage patterns
        2. Suggesting multi-functional space opportunities
        3. Recommending layout changes to improve efficiency
        4. Providing a functional map for the building
        """,
        expected_output="Detailed layout recommendations with specific actions",
        agent=agent
    )





def run_thermal_analysis(env_params):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate the report.")
        return None
    with st.spinner("Running thermal comfort analysis with CrewAI..."):
        thermal_agent = create_thermal_agent()
        thermal_task = create_thermal_analysis_task(env_params, thermal_agent)
        thermal_crew = Crew(
            agents=[thermal_agent],
            tasks=[thermal_task],
            verbose=True,
            process=Process.sequential
        )
        try:
            thermal_result = thermal_crew.kickoff()
            return thermal_result

        except Exception as e:
            st.error(f"Error running thermal analysis: {str(e)}")
            return None

def run_energy_optimization(energy_inputs, pricing_scheme):
    

    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None

    with st.spinner("Running energy optimization with CrewAI..."):
        # 1. Create the agent
        energy_agent = create_energy_agent()

        # 2. Create the task and extract forecast + reduction data
        task_data = create_energy_optimization_task(energy_inputs, energy_agent, pricing_scheme)
        task = task_data["task"]
        forecast_df = task_data["forecast_df"]
        reduction_result = task_data["reduction_result"]

        # 3. Run the Crew with this single task
        crew = Crew(
            agents=[energy_agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )

        try:
            result = crew.kickoff()

            return {
                "report": result,
                "forecast_series": forecast_df["forecast"],  # pd.Series
                "optimized_kw": reduction_result["estimated_reduction_kw"]  # float
            }

        except Exception as e:
            st.error(f"❌ Error running energy optimization: {str(e)}")
            return None

def run_multi_building_optimization(building_data_list, pricing_scheme):
    """
    Runs energy optimization for multiple buildings.

    Args:
        building_data_list (List[Dict]): List of energy_inputs for each building
        pricing_scheme (Dict): Time-based dynamic pricing scheme

    Returns:
        List[Dict]: List of results per building, including forecast, reduction, and markdown report
    """

    if not st.session_state.get("openai_api_key"):
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None

    all_results = []

    with st.spinner("Running energy optimization for all buildings..."):
        # Create a single energy agent to reuse
        energy_agent = create_energy_agent()

        # Call batch task creation
        task_outputs = create_multi_building_optimization_tasks(building_data_list, energy_agent, pricing_scheme)

        for output in task_outputs:
            idx = output["building_index"]

            if output.get("error"):
                all_results.append({
                    "building_index": idx,
                    "report": None,
                    "forecast_series": None,
                    "optimized_kw": None,
                    "error": output["error"]
                })
                continue

            task = output["task"]
            forecast_df = output["forecast_df"]
            reduction_result = output["reduction_result"]

            crew = Crew(
                agents=[energy_agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )

            try:
                result = crew.kickoff()

                all_results.append({
                    "building_index": idx,
                    "report": result,
                    "forecast_series": forecast_df["forecast"],
                    "optimized_kw": reduction_result["estimated_reduction_kw"],
                    "error": None
                })

            except Exception as e:
                all_results.append({
                    "building_index": idx,
                    "report": None,
                    "forecast_series": None,
                    "optimized_kw": None,
                    "error": str(e)
                })

    return all_results

def run_space_optimization(rooms_df):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None

    # Convert dataframe to string representation for the task
    rooms_data = rooms_df

    with st.spinner("Running space optimization analysis with CrewAI..."):
        space_agent = create_space_optimization_agent()
        space_task = create_space_optimization_task(rooms_data, space_agent)

        try:
            # Set environment variable for LiteLLM
            if 'OPENAI_API_KEY' not in os.environ and st.session_state.openai_api_key:
                os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key

            space_crew = Crew(
                agents=[space_agent],
                tasks=[space_task],
                verbose=True,
                process=Process.sequential
            )
            space_result = space_crew.kickoff()

            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": space_result})

            return space_result
        except Exception as e:
            error_msg = f"Error running space optimization: {str(e)}"
            st.error(error_msg)

            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return None


def run_layout_recommendation(rooms_df):
    if not st.session_state.openai_api_key:
        st.error("Please enter a valid OpenAI API key to generate recommendations.")
        return None
    
    # Convert dataframe to string representation for the task
    rooms_data = rooms_df.to_string()
    
    with st.spinner("Running layout recommendation analysis with CrewAI..."):
        layout_agent = create_layout_recommendation_agent()
        layout_task = create_layout_recommendation_task(rooms_data, layout_agent)
        layout_crew = Crew(
            agents=[layout_agent],
            tasks=[layout_task],
            verbose=True,
            process=Process.sequential
        )
        try:
            layout_result = layout_crew.kickoff()
            st.session_state.messages.append({"role": "assistant", "content": layout_result})
            return layout_result
        except Exception as e:
            st.error(f"Error running layout recommendation: {str(e)}")
            return None
        
def assign_tasks_dynamically(tasks, agents, llm):
    assigned_tasks = []
    for task in tasks:
        prompt = f"""
You are a Supervisor Agent. Assign the best-suited agent to this task.

Task:
\"\"\" 
{task.description} 
\"\"\"

Available agents:
{chr(10).join([f"- {agent.role}: {agent.goal}" for agent in agents])}

Respond ONLY with the agent role name.
"""
        response = llm.invoke(prompt)
        result = response.content.strip()

        # Match agent role by fuzzy string match
        matched_agent = next((a for a in agents if a.role.lower() in result.lower()), None)
        if matched_agent:
            task.agent = matched_agent
            assigned_tasks.append(task)
        else:
            print(f"⚠️ Could not assign task: {task.description}")
    return assigned_tasks


def chat_ui():
    st.divider()
    st.subheader("💬 Conversational Q&A with Agent")

    # Optional summary of available data
    available_contexts = []
    results = st.session_state.get("results", {})
    if results.get("thermal_analysis"):
        available_contexts.append("Thermal Analysis")
    if results.get("energy_optimization"):
        available_contexts.append("Energy Optimization")
    if results.get("space_optimization"):
        available_contexts.append("Space Optimization")
    if results.get("layout_recommendation"):
        available_contexts.append("Layout Recommendation")
    if results.get("unemployment_policy"):
        available_contexts.append("Unemployment Policy")

    if available_contexts:
        st.caption(f"📎 Available context: {', '.join(available_contexts)}")

    # Show conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input from user
    if prompt := st.chat_input("Ask a question about your analysis, layout, or job policy..."):
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_chat_query(prompt)  # This will also render charts/tables if matched
                if isinstance(response, str):
                    st.markdown(response)
                # If response is rendered via another function (e.g., chart/table), show nothing

        st.session_state.messages.append({"role": "assistant", "content": response if isinstance(response, str) else "[Graph/Table Rendered]"})

def generate_dynamic_thermal_template_custom(n_buildings, outdoor_temp):
    
    building_types = ["Office", "Residential", "Educational"]
    seasons = ["Summer", "Winter"]

    data = {
        'building_id': [],
        'air_temperature_celsius': [],
        'mean_radiant_temperature_celsius': [],
        'relative_humidity_percent': [],
        'air_velocity_meters_per_second': [],
        'metabolic_rate_met': [],
        'clothing_insulation_clo': [],
        'building_type': [],
        'season': [],
        'number_of_occupants': [],
        'outdoor_temperature_celsius': []  
    }

    for i in range(n_buildings):
        btype = random.choice(building_types)
        season = random.choice(seasons)

        if season == "Summer":
            tdb = round(random.uniform(25, 28), 1)
            tr = tdb + round(random.uniform(-0.5, 1.0), 1)
            rh = random.randint(50, 65)
            vr = round(random.uniform(0.15, 0.3), 2)
            clo = 0.5
        else:
            tdb = round(random.uniform(20, 23), 1)
            tr = tdb + round(random.uniform(-0.5, 0.5), 1)
            rh = random.randint(35, 50)
            vr = round(random.uniform(0.1, 0.2), 2)
            clo = 1.0

        if btype == "Office":
            met = 1.2
            occupants = random.randint(20, 100)
        elif btype == "Residential":
            met = 1.0
            occupants = random.randint(2, 10)
        elif btype == "Educational":
            met = 1.4
            occupants = random.randint(30, 100)

        data['building_id'].append(f"B{i+1}")
        data['air_temperature_celsius'].append(tdb)
        data['mean_radiant_temperature_celsius'].append(tr)
        data['relative_humidity_percent'].append(rh)
        data['air_velocity_meters_per_second'].append(vr)
        data['metabolic_rate_met'].append(met)
        data['clothing_insulation_clo'].append(clo)
        data['building_type'].append(btype)
        data['season'].append(season)
        data['number_of_occupants'].append(occupants)
        data['outdoor_temperature_celsius'].append(outdoor_temp)  

    return pd.DataFrame(data)


def generate_energy_template(num_buildings: int) -> pd.DataFrame:
    np.random.seed(42)  # For reproducibility

    data = {
        'current_temperature_celsius': np.round(np.random.uniform(22, 28, num_buildings), 1),
        'ambient_light_level_lux': np.random.randint(300, 800, num_buildings),
        'current_power_consumption_kilowatts': np.round(np.random.uniform(35, 60, num_buildings), 1),
        'energy_tariff_rate_per_kilowatt_hour': np.round(np.random.uniform(0.10, 0.25, num_buildings), 2),
        'lighting_power_usage_kilowatts': np.round(np.random.uniform(5, 20, num_buildings), 1),
        'appliance_power_usage_kilowatts': np.round(np.random.uniform(10, 25, num_buildings), 1),
        'building_area_square_meters': np.random.randint(50, 300, num_buildings),
        'setpoint_temperature_celsius': np.random.choice([21, 22, 23, 24], num_buildings),
        'occupancy_rate_decimal': np.round(np.random.uniform(0.5, 1.0, num_buildings), 2),
        'hvac_system_status': np.random.choice(['On - Cooling', 'On - Heating', 'Off'], num_buildings)
    }

    return pd.DataFrame(data)



def enforce_full_forms(text):
    # Normalize to uppercase first
    text = re.sub(r'\bpmv\b', 'PMV', text, flags=re.IGNORECASE)
    text = re.sub(r'\bppd\b', 'PPD', text, flags=re.IGNORECASE)
    text = re.sub(r'\butci\b', 'UTCI', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmet\b', 'MET', text, flags=re.IGNORECASE)
    text = re.sub(r'\bclo\b', 'CLO', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhvac\b', 'HVAC', text, flags=re.IGNORECASE)

    # Replace every instance with full form — always
    text = re.sub(r'\bPMV\b', 'PMV (Predicted Mean Vote)', text)
    text = re.sub(r'\bPPD\b', 'PPD (Predicted Percentage of Dissatisfied)', text)
    text = re.sub(r'\bUTCI\b', 'UTCI (Universal Thermal Climate Index)', text)
    text = re.sub(r'\bMET\b', 'MET (Metabolic Rate)', text)
    text = re.sub(r'\bCLO\b', 'CLO (Clothing Insulation)', text)
    text = re.sub(r'\bHVAC\b', 'HVAC (Heating, Ventilation, and Air Conditioning)', text)

    return text



def thermal_comfort_agent_ui():
    st.header("Thermal Comfort Analyst")
    st.caption("**Goal:** Analyze indoor environmental parameters and generate simple, friendly comfort reports")

    tab1, tab2 = st.tabs(["Single Building Analysis", "Multiple Buildings Analysis"])

    with tab1:
        st.subheader("Single Building Input")
        lat = st.number_input("Latitude", -90.0, 90.0, 20.0)
        lon = st.number_input("Longitude", -180.0, 180.0, 78.0)
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))

        weather_data = get_weather_data(lat, lon)
        if weather_data:
            st.success(f"Fetched weather: {weather_data['temperature_2m']}°C, RH {weather_data['relative_humidity_2m']}%, Wind {weather_data['wind_speed_10m']} m/s")

        building_type = st.selectbox("Building Type", ["Office", "Residential", "Educational"])
        season = st.selectbox("Season", ["Summer", "Winter"])
        num_people = st.number_input("Number of Occupants", 1, 100, 1)

        tdb = st.number_input("Air Temperature (°C)", value=weather_data['temperature_2m'] if weather_data else 23.0)
        tr = st.number_input("Mean Radiant Temperature (°C)", value=tdb)
        rh = st.slider("Relative Humidity (%)", 0, 100, int(weather_data['relative_humidity_2m']) if weather_data else 45)

        wind_speed = weather_data['wind_speed_10m'] if weather_data else 0.1
        capped_wind = min(wind_speed, 1.0)
        vr = st.number_input("Air Velocity (m/s)", 0.0, 1.0, value=capped_wind)

        met = st.select_slider("Activity Level (met)", [1.0, 1.2, 1.4, 1.6, 2.0], value=1.4)
        clo = st.select_slider("Clothing Insulation (clo)", [0.5, 0.7, 1.0, 1.5, 2.0], value=0.5)
        outdoor_temp = weather_data['temperature_2m'] if weather_data else tdb

        if st.button("Run Comfort Analysis"):
            if not st.session_state.openai_api_key:
                st.error("Please add your OpenAI API key")
            else:
                st.session_state.thermal_analysis = ""
                inputs = {
                    'tdb': tdb, 'tr': tr, 'rh': rh, 'vr': vr, 'met': met, 'clo': clo,
                    'building_type': building_type, 'season': season, 'num_people': num_people, 'outdoor_temp': outdoor_temp
                }
                result = run_thermal_analysis(inputs)
                st.session_state.thermal_analysis = result
                st.success("Comfort analysis complete!")

        if st.session_state.thermal_analysis:
            st.markdown("### 🔍 Comfort Analysis Result")
            st.markdown(st.session_state.thermal_analysis)         

    with tab2:
        st.subheader("Multiple Building Simulation")
        st.markdown("### 🌍 Location for Outdoor Weather Data")

        lat = st.number_input("Latitude", -90.0, 90.0, value=20.0, key="multi_lat")
        lon = st.number_input("Longitude", -180.0, 180.0, value=78.0, key="multi_lon")

        
        n_buildings = st.number_input("Number of Buildings", 1, 100, 3)
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))


        if 'multi_building_df' not in st.session_state:
            st.session_state.multi_building_df = None
            st.session_state.multi_building_result = None

        if st.button("Generate Building Data", key = "generate_button"):
            weather_data = get_weather_data(lat, lon)
            if not weather_data:
                st.error("❌ Failed to fetch outdoor weather data.")
            else:
                outdoor_temp = weather_data['temperature_2m']
                df = generate_dynamic_thermal_template_custom(n_buildings, outdoor_temp)
                st.session_state.multi_building_df = df
                st.session_state.multi_building_result = None
                st.success("✅ Building dataset generated.")

        if st.session_state.multi_building_df is not None:
            st.markdown("### 🏢 Generated Building Dataset")
            st.dataframe(st.session_state.multi_building_df)

            if st.button("Run Comfort Analysis", key = "run_analysis_button"):
                st.session_state_multi_building_result = None
                results, _ = run_multi_building_thermal_analysis(
                        st.session_state.multi_building_df
                )
                st.session_state.multi_building_result = results
                st.success("✅ Comfort analysis completed")

        if st.session_state.multi_building_result:
            st.markdown("### Combined Comfort Report")
            st.markdown(st.session_state.multi_building_result)
                    

    chat_ui()

def ensure_hourly_energy_series():
    """Ensure hourly energy usage data exists; simulate if missing."""
    if "hourly_energy_series" not in st.session_state:
        # Simulate historical energy usage (48 hours)
        now = datetime.now()
        index = pd.date_range(end=now, periods=48, freq='h')
        values = [round(random.uniform(38.0, 52.0), 2) for _ in range(48)]
        st.session_state["hourly_energy_series"] = pd.Series(values, index=index)  

def energy_optimization_agent_ui():
    st.header("Energy Optimization Engineer")
    st.caption("**Goal:** Recommend energy-saving actions while maintaining thermal comfort")

    tab1, tab2 = st.tabs(["Single Building Analysis", "Multi-Building Analysis"])

    with tab1:
        st.subheader("Energy Consumption Parameters")

        col1, col2 = st.columns(2)
        with col1:
            building_area = st.number_input("Building Area (m²)", min_value=10.0, max_value=10000.0, value=100.0, step=10.0)
            current_temperature = st.number_input("Current Indoor Temperature (°C)", min_value=10.0, max_value=40.0, value=23.0, step=0.5)
            setpoint_temp = st.number_input("HVAC Setpoint Temperature (°C)", min_value=15.0, max_value=30.0, value=22.0, step=0.5)
            ambient_light_level = st.number_input("Ambient Light Level (Lux)", min_value=0, max_value=2000, value=300, step=50)
            occupancy_rate = st.slider("Occupancy Rate", 0.0, 1.0, value=0.7, step=0.1)

        with col2:
            current_power_consumption = st.number_input("Current Total Power Consumption (kW)", min_value=0.0, max_value=1000.0, value=45.0, step=1.0)
            lighting_power_usage = st.number_input("Lighting Power Usage (kW)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
            appliance_power_usage = st.number_input("Appliance Power Usage (kW)", min_value=0.0, max_value=500.0, value=18.0, step=1.0)
            energy_tariff_rate = st.number_input("Energy Tariff Rate ($/kWh)", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
            hvac_status = st.selectbox("HVAC Status", ["On - Cooling", "On - Heating", "Off", "Auto"])

        if st.button("Execute Energy Optimization Analysis"):
            if not st.session_state.get("openai_api_key"):
                st.error("Please enter a valid OpenAI API key.")
                return
            
            if "hourly_energy_series" not in st.session_state:
                st.session_state.hourly_energy_series = generate_synthetic_energy_series(current_power_consumption)

            ensure_hourly_energy_series()

            # Prepare input dictionary
            energy_inputs = {
                'building_area': building_area,
                'current_temperature': current_temperature,
                'setpoint_temp': setpoint_temp,
                'ambient_light_level': ambient_light_level,
                'occupancy_rate': occupancy_rate,
                'current_power_consumption': current_power_consumption,
                'lighting_power_usage': lighting_power_usage,
                'appliance_power_usage': appliance_power_usage,
                'energy_tariff_rate': energy_tariff_rate,
                'hvac_status': hvac_status,
                'lighting_power_density': 10  # Fixed value; optional for user to control
            }

            pricing_scheme = {
                0: 0.10, 1: 0.10, 17: 0.30, 18: 0.35, 19: 0.33,
                'default': 0.15
            }

            # Run optimization
            result = run_energy_optimization(energy_inputs, pricing_scheme)

            if result:
                st.session_state.results["energy_optimization"] = result["report"]

                st.subheader("📘 Agent Recommendation")
                st.markdown(result["report"])

                # ✅ Plot forecast vs optimized usage
                st.subheader("📉 Energy Forecast vs Optimization")
                plot_actual_vs_optimized(result["forecast_series"], result["optimized_kw"])
            else:
                st.error("❌ Optimization task failed.")

        with tab2:
            st.subheader("Multi-Building Energy Analysis")
            st.info("Upload a CSV file with energy records for batch optimization analysis.")

    # Let user define number of buildings
            num_buildings = st.number_input("Number of buildings to generate in template", min_value=1, max_value=100, value=3)
    
            if st.button("Generate Dynamic CSV Template"):
                template_df = generate_energy_template(num_buildings)  # Uses the earlier helper function
                st.download_button(
            label="📥 Download Energy CSV Template",
            data=template_df.to_csv(index=False),
            file_name="energy_optimization_template.csv",
            mime="text/csv"
                )

            uploaded_file = st.file_uploader("Upload CSV file with building energy data", type="csv")

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)

                st.success(f"Successfully loaded {len(df)} building records")
                st.dataframe(df, use_container_width=True)

                if st.button("Execute Batch Energy Optimization"):
                    if not st.session_state.get("openai_api_key"):
                        st.error("Please enter a valid OpenAI API key.")
                    else:
                        reports = []
                        forecasts = []

                        pricing_scheme = {
                    0: 0.10, 1: 0.10, 17: 0.30, 18: 0.35, 19: 0.33,
                    'default': 0.15
                    }

                        for i, row in df.iterrows():
                            energy_inputs = {
                        'building_area': row['building_area_square_meters'],
                        'current_temperature': row['current_temperature_celsius'],
                        'setpoint_temp': row['setpoint_temperature_celsius'],
                        'ambient_light_level': row['ambient_light_level_lux'],
                        'occupancy_rate': row['occupancy_rate_decimal'],
                        'current_power_consumption': row['current_power_consumption_kilowatts'],
                        'lighting_power_usage': row['lighting_power_usage_kilowatts'],
                        'appliance_power_usage': row['appliance_power_usage_kilowatts'],
                        'energy_tariff_rate': row['energy_tariff_rate_per_kilowatt_hour'],
                        'hvac_status': row['hvac_system_status'],
                        'lighting_power_density': 10  # Optional fixed value
                    }

                            synthetic_series = generate_synthetic_energy_series(
                        row['current_power_consumption_kilowatts']
                    )
                            st.session_state.hourly_energy_series = synthetic_series
                            ensure_hourly_energy_series()

                            result = run_energy_optimization(energy_inputs, pricing_scheme)

                            if result:
                                reports.append((i, result["report"]))
                                forecasts.append((i, result["forecast_series"], result["optimized_kw"]))
                            else:
                                reports.append((i, f"❌ Failed to process building index {i}"))

                        st.subheader("📘 Optimization Reports")
                        for idx, report in reports:
                            st.markdown(f"### 🏢 Building {idx + 1}")
                            st.markdown(report)

                        st.subheader("📉 Forecast vs Optimized Energy Usage")
                        for idx, forecast_series, optimized_kw in forecasts:
                            st.markdown(f"### 🔋 Building {idx + 1}")
                            plot_actual_vs_optimized(forecast_series, optimized_kw)

    chat_ui()


def space_optimization_agent_ui():
    st.header("🏢 Space Optimization Assistant")
    st.caption("**Goal:** Optimize room allocation using occupancy and environmental data")

    st.subheader("1️⃣ Generate Room Dataset")

    # Input: number of rooms to simulate
    num_rooms = st.number_input(
        "Enter number of rooms to simulate:", min_value=1, max_value=100, value=5, step=1
    )

    # Button to generate room data
    if st.button("🔄 Generate Room Data"):
        st.session_state.rooms_df = get_live_room_data(num_rooms=num_rooms)
        st.success(f"✅ Generated dataset for {num_rooms} rooms.")

    # Proceed if room data is available
    if "rooms_df" in st.session_state and st.session_state.rooms_df is not None:
        rooms_df = st.session_state.rooms_df

        st.subheader("📊 Room Data Preview")
        st.dataframe(rooms_df, use_container_width=True)

        st.subheader("2️⃣ Run Space Optimization Analysis")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Run Optimization Analysis"):
                if not st.session_state.openai_api_key:
                    st.error("Please enter a valid OpenAI API key.")
                else:
                    space_result = run_space_optimization(rooms_df)
                    if space_result:
                        st.session_state.results["space_optimization"] = space_result
                        st.success("✅ Optimization analysis complete.")

        with col2:
            if st.button("🔁 Reset Session"):
                st.session_state.clear()
                st.rerun()

        # Visualization of occupancy
        st.subheader("📉 Room Occupancy Visualization")
        fig = {
            'data': [
                {
                    'x': rooms_df['RoomID'],
                    'y': rooms_df['Occupancy'],
                    'type': 'bar',
                    'marker': {
                        'color': rooms_df['Occupancy'].apply(
                            lambda x: 'red' if x > 0.7 else 'orange' if x > 0.4 else 'green'
                        )
                    }
                }
            ],
            'layout': {
                'title': 'Room Occupancy Levels',
                'xaxis': {'title': 'Room ID'},
                'yaxis': {'title': 'Occupancy Rate (0-1)'}
            }
        }
        st.plotly_chart(fig, use_container_width=True)

        # Show optimization result if available
        if "space_optimization" in st.session_state.results:
            st.subheader("📋 Optimization Results")
            st.markdown(st.session_state.results["space_optimization"])

    # Optional chatbot / interaction panel
    chat_ui()

def show_optimized_layout_plotly(rooms_df, recommendations):
    # Define color mapping for each type of room
    color_map = {
        "Meeting Room": "#87CEFA",          # Light Blue
        "Focus Pod": "#90EE90",             # Light Green
        "Storage or Flex Space": "#FFDAB9", # Peach
        "Storage": "#FFB6C1",               # Pink (fallback)
        "Multi-use Room": "#FFFACD"         # Lemon
    }

    # Unique list of categories for vertical axis
    categories = sorted(set(recommendations.values()))
    category_y = {cat: i for i, cat in enumerate(categories[::-1])}

    fig = go.Figure()

    # Arrange rooms in rows (y = category) and columns (x = position in category)
    category_counts = {cat: 0 for cat in categories}

    for room_id, function in recommendations.items():
        x = category_counts[function] + 1
        y = category_y.get(function, 0)

        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            text=f"{room_id}<br>{function}",
            textposition="middle center",
            marker=dict(
                symbol='square',
                size=90,
                color=color_map.get(function, "#D3D3D3"),
                line=dict(color='black', width=1)
            ),
            hovertemplate=f"<b>{room_id}</b><br>Type: {function}<extra></extra>"
        ))

        category_counts[function] += 1

    fig.update_layout(
        title="📐 Optimized Room Layout Map",
        xaxis=dict(visible=False),
        yaxis=dict(
            tickvals=list(category_y.values()),
            ticktext=list(category_y.keys()),
            showgrid=False,
            zeroline=False
        ),
        plot_bgcolor="white",
        height=max(400, 120 * len(categories)),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

def layout_recommendation_agent_ui():
    st.header("Layout Planner")
    st.caption("**Goal:** Analyze room data and suggest long-term layout changes for flexibility and efficiency")

    # Generate or use session state room data
    if "rooms_df" not in st.session_state or st.session_state.get("num_rooms") != st.session_state.num_rooms:
        st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)

    rooms_df = st.session_state.rooms_df

    st.subheader("Current Layout Data")
    st.dataframe(rooms_df, use_container_width=True)

    layout_result = None
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Refresh Layout Data"):
            st.session_state.rooms_df = get_live_room_data(num_rooms=st.session_state.num_rooms)
            st.rerun()

    with col2:
        if st.button("Generate Layout Recommendations"):
            if not st.session_state.openai_api_key:
                st.error("Please enter a valid OpenAI API key to generate recommendations.")
            else:
                layout_result = run_layout_recommendation(rooms_df)
                if layout_result:
                    st.session_state.results["layout_recommendation"] = layout_result
                    st.success("✅ Layout recommendation generated!")

    # 📄 Show text recommendations
    if "layout_recommendation" in st.session_state.results:
        layout_result = st.session_state.results["layout_recommendation"]
        st.subheader("🧠 Layout Recommendations (Textual)")
        st.markdown(layout_result)

        # 🧩 Improved parsing: extract room-function mapping
        recommendations = {}
        for line in layout_result.splitlines():
            matches = re.findall(r"(Room[_ ]?\d+)[^\n]*?(Focus Pod|Meeting Room|Multi-use Room|Storage or Flex Space|Storage)", layout_result, re.IGNORECASE)
            for room_id, func in matches:
                room_id = room_id.replace(" ", "_").strip()
                if "Storage" in func and "Flex" in func:
                    function = "Storage or Flex Space"
                elif "Storage" in func:
                    function = "Storage or Flex Space"
                elif "Focus Pod" in func:
                    function = "Focus Pod"
                elif "Meeting" in func:
                    function = "Meeting Room"
                elif "Multi-use" in func:
                    function = "Multi-use Room"
                else:
                    function = "Multi-use Room"
                recommendations[room_id] = function


        # 📊 Show Plotly layout if parsed successfully
        if recommendations:
            st.subheader("📐 Optimized Room Layout (Interactive)")
            show_optimized_layout_plotly(rooms_df, recommendations)

    # 📊 Power usage chart
    st.subheader("Room Usage Analysis")
    fig1 = {
        'data': [
            {
                'x': rooms_df['RoomID'],
                'y': rooms_df['Power_kWh'],
                'type': 'bar',
                'name': 'Power Usage (kWh)',
                'marker': {'color': 'blue'}
            }
        ],
        'layout': {
            'title': 'Power Usage by Room',
            'xaxis': {'title': 'Room ID'},
            'yaxis': {'title': 'Power (kWh)'}
        }
    }
    st.plotly_chart(fig1, use_container_width=True)

    # 📊 Radar chart for room metrics
    fig2 = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': [
                    row['Occupancy'],
                    row['Temperature'] / 30,
                    row['CO2'] / 1200,
                    row['Light'] / 600,
                    row['Humidity'] / 100,
                    row['Power_kWh'] / 4
                ],
                'theta': ['Occupancy', 'Temperature', 'CO2', 'Light', 'Humidity', 'Power'],
                'fill': 'toself',
                'name': row['RoomID']
            } for _, row in rooms_df.iterrows()
        ],
        'layout': {
            'title': 'Room Metrics Comparison',
            'polar': {'radialaxis': {'visible': True, 'range': [0, 1]}}
        }
    }
    st.plotly_chart(fig2, use_container_width=True)

    # 🧠 Enable conversational follow-up
    chat_ui()




def unemployment_policy_agent_ui():
    st.header("Unemployment Policy Agent")
    st.caption("**Goal:** Analyze job market trends and unemployment disparities to guide citizens toward suitable employment and training programs.")

    # 🔐 Check and set API key
    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to run the agent.")
        return

    supervisor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=api_key)

    st.subheader("Input Parameters")

    # 🔹 Free-text input for Region
    region = st.text_input("Enter Region (e.g., California, Texas)", placeholder="Type a region")

    # 🔹 Free-text input for Skills
    skill_text = st.text_input("Enter Skills (comma-separated)", placeholder="e.g., Python, Excel, SQL")
    profile_skills = [skill.strip() for skill in skill_text.split(",") if skill.strip()]

    # 🔹 Other inputs
    experience_years = st.slider("Years of Experience", 0, 10, 1)
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's degree", "Master's degree"])
    desired_title = st.text_input("Desired Job Title")

    linkedin_url = st.text_input("LinkedIn Profile URL (Optional)", placeholder="https://www.linkedin.com/in/your-name")
    summary_text = ""
    if not linkedin_url:
        summary_text = st.text_area("Summary of Experience", placeholder="If no LinkedIn URL is provided, describe your experience here...")

    user_profile = {
        "skills": profile_skills,
        "experience_years": experience_years,
        "education": education_level,
        "location": region,
        "desired_title": desired_title,
        "linkedin_url": linkedin_url,
        "summary": summary_text
    }

    st.session_state["public_sector_profile"] = user_profile
    st.session_state["public_sector_region"] = region

    st.subheader("Interact with Agent")
    if st.button("Execute Unemployment Analysis"):
        with st.spinner("Running Unemployment Policy Agent Crew..."):
            # Dynamically create agents based on inputs
            job_market_agent = Agent(
                role="Job Market Monitoring Agent",
                goal=f"Collect live job posting data for {region} from platforms like Indeed and LinkedIn, and identify the top 5 high-demand roles.",
                backstory="I uncover hiring trends by scraping job boards.",
                allow_delegation=True,
                verbose=True,
                tools=[search_tool, scrape_tool]
            )

            demographic_agent = Agent(
                role="Demographic Impact Agent",
                goal=f"Fetch unemployment rate data by age, education, and ethnicity for {region}, analyze disparities, and highlight the most affected groups.",
                backstory="I analyze public statistics to assess unemployment disparities.",
                allow_delegation=True,
                verbose=True,
                tools=[search_tool, scrape_tool]
            )

            citizen_agent = Agent(
                role="Citizen Guidance Agent",
                goal=(
                    f"Based on job market and demographic findings, provide personalized guidance "
                    f"on suitable jobs and retraining programs for a user with skills: {', '.join(profile_skills)}, "
                    f"experience: {experience_years} years, education: {education_level}, location: {region}, "
                    f"desired role: {desired_title}. {f'LinkedIn: {linkedin_url}' if linkedin_url else f'Summary: {summary_text}'}"
                ),
                backstory="I craft actionable recommendations tailored to individual needs.",
                allow_delegation=True,
                verbose=True,
                tools=[search_tool, scrape_tool]
            )

            # Dynamically create tasks
            job_task = Task(
                description=f"Monitor live job postings for {region} and identify the top 5 high-demand roles.",
                expected_output="Ranked list of the top 5 high-demand roles with summary statistics.",
                agent=None
            )

            demo_task = Task(
                description=f"Retrieve and analyze unemployment rates by age, education, and ethnicity for {region}. Highlight the most affected groups.",
                expected_output="Report on demographic disparities in unemployment.",
                agent=None
            )

            guidance_task = Task(
                description=(
                    f"Based on the following user profile, job market, and demographic trends, generate personalized job and training recommendations.\n\n"
                    f"User Details:\n"
                    f"- Skills: {', '.join(user_profile['skills'])}\n"
                    f"- Experience: {user_profile['experience_years']} years\n"
                    f"- Education: {user_profile['education']}\n"
                    f"- Location: {user_profile['location']}\n"
                    f"- Desired Job Title: {user_profile['desired_title']}\n"
                    f"{'Refer to their LinkedIn profile for detailed experience: ' + user_profile['linkedin_url'] if user_profile['linkedin_url'] else 'Summary of experience: ' + user_profile['summary']}\n\n"
                    f"Use this data to tailor recommendations to specific industries, training, or roles the user can explore. Highlight why each suggestion matches the profile."
            ),
                expected_output="Tailored job and retraining recommendations based on the user's background and job market trends.",
                agent=None
            )


            agents = [citizen_agent, demographic_agent, job_market_agent]
            tasks = [job_task, demo_task, guidance_task]
            assigned_tasks = assign_tasks_dynamically(tasks, agents, supervisor_llm)

            crew = Crew(
                agents=agents,
                tasks=assigned_tasks,
                verbose=True,
                process=Process.hierarchical,
                manager_llm=supervisor_llm
            )
            result = crew.kickoff()

            if result:
                st.session_state.results["unemployment_policy"] = result
                st.success("✅ Analysis complete.")
                st.subheader("📋 AI-Generated Report")
                st.markdown(result)

    chat_ui()


def healthcare_policy_agent_ui():
    st.header("Healthcare Policy Agent")
    st.caption("**Goal:** Monitor healthcare policy changes and guide users toward eligible programs and local care resources.")

    # 🔐 Check and set API key
    api_key = st.session_state.get("openai_api_key", "")
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to run the agent.")
        return

    supervisor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=api_key)

    st.subheader("Input Parameters")

    # 🔹 User Profile Input
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=40)
        income = st.number_input("Annual Income ($)", min_value=0, value=30000)
        family_size = st.number_input("Family Size", min_value=1, value=3)

    with col2:
        current_insurance = st.selectbox("Current Insurance", ["None", "Medicaid", "Marketplace", "Employer-based", "Private"])
        location = st.text_input("Location (City)", placeholder="e.g., Los Angeles")

    # 🧾 Store user profile
    user_profile = {
        "age": age,
        "income": income,
        "family_size": family_size,
        "insurance": current_insurance,
        "location": location
    }

    st.session_state["healthcare_profile"] = user_profile
    st.session_state["healthcare_location"] = location

    st.subheader("Interact with Agent")
    if st.button("Execute Healthcare Policy Analysis"):
        with st.spinner("Running Healthcare Policy Agent Crew..."):

            # === Define Agents ===
            policy_update_agent = Agent(
                role="Policy Update Monitoring Agent",
                goal=f"Scrape and summarize recent Medicaid, ACA, CHIP, and public health policy changes relevant to {location}.",
                backstory="I track official sites and identify new or updated healthcare policies.",
                allow_delegation=True,
                verbose=True,
                tools=[search_tool, scrape_tool]
            )

            eligibility_agent = Agent(
                role="Eligibility Analysis Agent",
                goal=(f"Evaluate whether the user is newly eligible for any health programs based on age {age}, income ${income}, "
                      f"family size {family_size}, and insurance status '{current_insurance}'."),
                backstory="I analyze health policies against the user's demographic and financial profile.",
                allow_delegation=True,
                verbose=True,
                tools=[search_tool, scrape_tool]
            )

            advisor_agent = Agent(
                role="Healthcare Access Guidance Agent",
                goal=(f"Based on policy updates and eligibility results, recommend specific actions the user in {location} should take: "
                      f"clinics to visit, websites to enroll, and steps to apply."),
                backstory="I advise citizens how to access care or benefits step-by-step.",
                allow_delegation=True,
                verbose=True,
                tools=[search_tool, scrape_tool]
            )

            # === Define Tasks ===
            policy_task = Task(
                description=(f"Search https://www.cms.gov, https://dhcs.ca.gov, and local government sites for policy changes in the past 7 days "
                             f"affecting Medicaid, ACA, CHIP, or healthcare subsidies relevant to {location}. "
                             f"Return a table with: Policy Title, Effective Date, Government Level, Summary."),
                expected_output="List of new healthcare policies with date, name, level, and summary.",
                agent=None
            )

            eligibility_task = Task(
                description=(
                    f"Based on the user's profile:\n"
                    f"- Age: {age}\n"
                    f"- Income: ${income}\n"
                    f"- Family Size: {family_size}\n"
                    f"- Current Insurance: {current_insurance}\n\n"
                    f"Cross-reference this with current policies and return a list of eligible programs with:\n"
                    f"- Program Name\n"
                    f"- Reason for Eligibility\n"
                    f"- Enrollment Deadline (if available)"
                ),
                expected_output="Bullet-point list of eligible programs with justification and deadlines.",
                agent=None
            )

            care_task = Task(
                description=(f"Based on the programs the user is eligible for, recommend actions specific to {location}:\n"
                             f"Enrollment links, public health clinics, contacts, and step-by-step guides for sign-up."),
                expected_output="List of recommended actions: location, contact, URL, and enrollment steps.",
                agent=None
            )

            agents = [policy_update_agent, eligibility_agent, advisor_agent]
            tasks = [policy_task, eligibility_task, care_task]
            assigned_tasks = assign_tasks_dynamically(tasks, agents, supervisor_llm)

            crew = Crew(
                agents=agents,
                tasks=assigned_tasks,
                verbose=True,
                process=Process.hierarchical,
                manager_llm=supervisor_llm
            )

            result = crew.kickoff()

            if result:
                st.session_state.results["healthcare_policy"] = result
                st.success("✅ Healthcare policy analysis complete.")
                st.subheader("📋 AI-Generated Report")
                st.markdown(result)

    chat_ui()




def show_agent_tools_table(use_case: Optional[str] = None):
    industry = st.session_state.get("industry", "")
    
    if not use_case or use_case == "Select Agent":
        st.subheader("Agent Configuration")
        if industry == "Public Sector":
            st.markdown("Showing agent tools for **Public Sector** (default)")
            st.table(tools_df_public_sector)
  # 👈 this is fallback
        else:
            st.table(tools_df)
    else:
        table_data = agent_tools_table_data.get(use_case, [])  # ✅ Make sure Healthcare Policy is included here
        if table_data:
            df = pd.DataFrame(table_data)
            st.subheader(f"{use_case}: Agents and Tools Used")
            st.table(df)

def show_agent_llm_table(use_case: Optional[str] = None):
    if not use_case or use_case == "Select Agent":
        st.subheader("LLM used in this Agents")
        industry = st.session_state.get("industry", "")
        if industry == "Public Sector":
            st.table(default_llm_config_public)
        else:
            st.table(default_llm_config)
    else:
        table_data = agent_llm_table_data.get(use_case, [])  # ✅ Must include Healthcare Policy here
        if table_data:
            df = pd.DataFrame(table_data)
            st.subheader(f"LLM used in {use_case}")
            st.table(df)



def generate_synthetic_energy_series(current_kw: float, hours: int = 24) -> pd.Series:
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    time_index = pd.date_range(end=now, periods=hours, freq='H')

    # Mild sinusoidal + noise pattern
    base_load = current_kw
    hour_offsets = np.linspace(-1, 1, hours)
    daily_pattern = 1 + 0.15 * np.sin(2 * np.pi * hour_offsets)
    noise = np.random.normal(loc=0, scale=0.02, size=hours)

    values = base_load * daily_pattern + base_load * noise
    values = np.clip(values, 0, None)

    return pd.Series(values, index=time_index, name="synthetic_energy_kw")



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

# Define tools for agents
@tool
def get_building_summary() -> str:
    """Return average environmental and occupancy statistics for the building."""
    return _get_building_summary_internal()




@tool
def get_weather_summary(city: str) -> str:
    """Return current weather conditions for a given city using Open-Meteo API."""
    return _get_weather_summary_internal(city)


def plot_actual_vs_optimized(forecast_series: pd.Series, optimized_kw: float):
    """
    Plot actual vs optimized energy usage based on forecast and predicted savings.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    # Construct optimized series
    optimized_series = forecast_series - optimized_kw
    optimized_series = optimized_series.clip(lower=0)

    df = pd.DataFrame({
        "Forecasted Energy (kW)": forecast_series,
        "Optimized Energy (kW)": optimized_series
    })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["Optimized Energy (kW)"], marker='o', linestyle='--', label="Optimized Forecast")
    ax.set_title("Actual vs Optimized Energy Forecast (Next 6 Hours)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Usage (kW)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)


def forecast_energy_usage_arima(energy_series: pd.Series, steps: int = 6, plot: bool = False) -> pd.DataFrame:
    """
    Forecast energy usage for next N hours using ARIMA.
    Returns a DataFrame with index as datetime and column "forecast".
    """
    energy_series = energy_series.sort_index().asfreq('h')

    model = ARIMA(energy_series, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    forecast_index = pd.date_range(start=energy_series.index[-1] + pd.Timedelta(hours=1), periods=steps, freq='h')
    forecast_df = pd.DataFrame({'forecast': forecast.values}, index=forecast_index)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(energy_series[-24:], label="Historical Usage", color="blue")
        plt.plot(forecast_df, label="Forecast", linestyle='--', marker='o', color="orange")
        plt.title("ARIMA Forecast")
        plt.xlabel("Time")
        plt.ylabel("Energy (kW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return forecast_df

def predict_energy_reduction_regression(features: dict) -> dict:
    """
    Simulate a regression model to predict energy reduction potential based on building features.

    Parameters:
    - features (dict): Building conditions and usage metrics

    Returns:
    - dict with predicted optimized energy and estimated reduction
    """
    # Feature vector (same order as training)
    X = np.array([[
        features['current_power_consumption'],
        features['occupancy_rate'],
        features['current_temperature'],
        features['setpoint_temp'],
        features['ambient_light_level'],
        features['lighting_power_usage'],
        features['appliance_power_usage'],
    ]])

    # Simulated trained regression model
    model = LinearRegression()
    model.coef_ = np.array([0.6, 5, 1.2, -1.5, 0.02, 0.8, 0.5])
    model.intercept_ = 5
    model._residues = 0  # compatibility fix

    # Actual values
    current_power = features['current_power_consumption']

    # Predict optimized power
    predicted_optimized_power = model.predict(X)[0]
    predicted_optimized_power = min(current_power, max(0, predicted_optimized_power))

    # Calculate reduction
    reduction = current_power - predicted_optimized_power
    percent_reduction = (reduction / current_power) * 100 if current_power > 0 else 0

    return {
        "current_power_kw": round(current_power, 2),
        "predicted_optimized_power_kw": round(predicted_optimized_power, 2),
        "estimated_reduction_kw": round(reduction, 2),
        "reduction_percent": round(percent_reduction, 1)
    }



def calculate_dynamic_pricing_cost(usage_series: pd.Series, pricing_scheme: dict, alert_threshold: float = 0.25) -> dict:
    """
    Calculate total energy cost using dynamic pricing and trigger alerts for high-tariff usage.

    Parameters:
    - usage_series (pd.Series): Hourly power usage in kWh, indexed by datetime.
    - pricing_scheme (dict): Dictionary mapping hour (0-23) to price in $/kWh.
                             Should include a 'default' key as fallback.
                             Example: {0: 0.10, 17: 0.30, 18: 0.35, 'default': 0.15}
    - alert_threshold (float): Tariff threshold to flag high-cost hours.

    Returns:
    - dict with:
        - total_cost: float
        - hourly_costs: pd.Series
        - high_cost_hours: list of (timestamp, usage, rate)
    """
    total_cost = 0.0
    hourly_costs = []
    high_cost_hours = []

    for timestamp, usage in usage_series.items():
        hour = timestamp.hour
        rate = pricing_scheme.get(hour, pricing_scheme.get("default", 0.15))
        cost = usage * rate
        total_cost += cost
        hourly_costs.append(cost)

        if rate >= alert_threshold and usage > 0:
            high_cost_hours.append((timestamp, usage, rate))

    hourly_cost_series = pd.Series(hourly_costs, index=usage_series.index)

    return {
        "total_cost": round(total_cost, 2),
        "hourly_costs": hourly_cost_series,
        "high_cost_hours": high_cost_hours
    }

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
        # Since we don't have the actual model file, we'll simulate the prediction
        current_energy = room.get('Power_kWh', 2.0)
        
        # Calculate optimized energy based on room parameters
        optimized_energy = current_energy * 0.7  # Assume 30% reduction
        
        # Calculate savings
        reduction = current_energy - optimized_energy
        percent = (reduction / current_energy) * 100

        return (
            f"🔋 Current Energy Usage: {current_energy:.2f} kWh\n"
            f"⚙️ Optimized Energy Usage: {optimized_energy:.2f} kWh\n"
            f"📉 Estimated Reduction: {reduction:.2f} kWh ({percent:.1f}%)"
        )
    except Exception as e:
        return f"❌ Failed to simulate energy reduction: {e}"
    



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
        return "\n".join(suggestions)
    except Exception as e:
        return f"❌ Error during consolidation suggestion: {e}"
    




def suggest_space_rezoning() -> str:
    """Cluster rooms into zones based on occupancy, temperature, CO2, light, humidity, and energy usage."""
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
        summary = "\n".join([f"Zone {z}: {count} room(s)" for z, count in cluster_counts.items()])
        return f"🧠 Re-zoning complete using KMeans clustering:\n{summary}"
    except Exception as e:
        return f"❌ Error during re-zoning: {e}"
    




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


        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

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

tool_function_map = {
    "Get Building Summary": get_building_summary,
    "Get Weather Summary": get_weather_summary,
    "Simulate Energy Reduction": simulate_energy_reduction,
    "Explain Energy Forecast": explain_energy_forecast_llm,
    "Predict Occupancy Trend": predict_occupancy_trend,
    "Suggest Room Consolidation": suggest_room_consolidation,
    "Suggest Space Rezoning": suggest_space_rezoning,
    "Recommend Layout Plan": recommend_layout_plan,
    "Identify Multiuse Zones": identify_multiuse_zones,
    "Room Function Mapping": recommend_room_function_map
}




def main():
    industry = st.session_state.get("industry", "")
    use_case = st.session_state.get("use_case", "")

    if not industry:
        st.info("👋 Please select an industry from the sidebar to begin.")
        return

    # ✅ Show agent tool/LLM tables for selected industry (even before use case)
    show_agent_tools_table(use_case)
    show_agent_llm_table(use_case)

    if not use_case:
        st.info("📌 Now select a use case to configure agent behavior.")
        return

    # ✅ If both are selected, render use-case-specific UIs
    if industry == "Energy Management in Buildings":
        if use_case == "Thermal Comfort Analysis":
            thermal_comfort_agent_ui()
        elif use_case == "Energy Optimization":
            energy_optimization_agent_ui()
        elif use_case == "Space Optimization":
            space_optimization_agent_ui()
        elif use_case == "Layout Planning":
            layout_recommendation_agent_ui()

    elif industry == "Public Sector":
        if use_case == "Unemployment Policy":
            unemployment_policy_agent_ui()
        elif use_case == "Healthcare Policy":
            healthcare_policy_agent_ui()

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
