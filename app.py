import os
import streamlit as st
from langchain.llms import AzureOpenAI
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType

weather = OpenWeatherMapAPIWrapper()

st.title('🏃AIWear')
st.subheader('What to wear for a run right now in...')
prompt = st.text_input('Location')


llm = AzureOpenAI(
    deployment_name="davinci",
    model_name="text-davinci-003",
    temperature=0.8
)

location_template = PromptTemplate(
    input_variables = ['weather_data'],
    template = 'What should I wear for a run in these conditions? {weather_data}. List the suggested clothes in a markdown formatted bullet list, separating upper and lower body. Do not list multiple clothes that occupy the same slot, like shorts and pants. Do not mention shoes.'
)

weather_chain = LLMChain(llm=llm, prompt=location_template, verbose=True)

if prompt:
    weather_data = weather.run(prompt)
    print(weather_data)
    if weather_data:
        response = weather_chain.run(weather_data=weather_data)
        st.write(response)
        with st.expander("Weather data"):
            st.write(weather_data)
