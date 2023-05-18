import os
import streamlit as st
import re
from langchain.llms import AzureOpenAI
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streamlit_js_eval import get_geolocation


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
    with st.spinner("Getting weather..."):
        weather_data = weather.run(prompt)
    print(weather_data)
    if weather_data:
        wind_match = re.search(r'Wind speed: ([\d.]+ m/s)', weather_data)
        status_match = re.search(r'Detailed status: ([\w\s]+?)\n', weather_data)
        temp_match = re.search(r'Current: ([\d.]+°C)', weather_data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Temperature", temp_match.group(1))
        col2.metric("Weather", status_match.group(1)) 
        col3.metric("Wind", wind_match.group(1)) 

        with st.spinner("Thinking..."):
            response = weather_chain.run(weather_data=weather_data)
        st.success("The AI suggests you should wear:")
        st.write(response)
        with st.expander("Weather data"):
            st.write(weather_data)
    else:
        st.write("Something went wrong")
