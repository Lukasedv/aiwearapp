import os
from keys import secrets
import streamlit as st
from langchain.llms import AzureOpenAI
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType

os.environ['OPENAI_API_TYPE'] = secrets.get('OPENAI_API_TYPE')
os.environ['OPENAI_API_VERSION'] = secrets.get('OPENAI_API_VERSION')
os.environ['OPENAI_API_BASE'] = secrets.get('OPENAI_API_BASE')
os.environ['OPENAI_API_KEY'] = secrets.get('OPENAI_API_KEY')
os.environ["OPENWEATHERMAP_API_KEY"] = secrets.get('OPENWEATHERMAP_API_KEY')

weather = OpenWeatherMapAPIWrapper()

st.title('🏃AIWear - What to wear for a run in...')
prompt = st.text_input('Location')

llm = AzureOpenAI(
    deployment_name="davinci",
    model_name="text-davinci-003", 
)

location_template = PromptTemplate(
    input_variables = ['weather_data'],
    template = 'What should I wear for a run in these conditions? {weather_data}. Include temperature and relevant weather data in the start of the respose like: "Helsinki, 13c, rain".'
)

weather_chain = LLMChain(llm=llm, prompt=location_template, verbose=True)

if prompt:
    weather_data = weather.run(prompt)
    print(weather_data)
    if weather_data:
        response = weather_chain.run(weather_data=weather_data)
        st.write(response)
