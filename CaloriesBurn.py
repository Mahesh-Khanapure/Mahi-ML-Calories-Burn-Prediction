import streamlit as st
import pickle
import numpy as np
import pandas as pd
html_temp="""
<div style="background-color:lightblue;padding:16px">
<h2 style="color:back;text-algin:center;">Calories Burn Prediction Using ML</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)

Model=pickle.load(open('Calories.pkl','rb'))

#Gender
p1=st.selectbox('Gender',['male','female'])
#Age
p2=st.slider('Age',1,100)
#Height
p3=st.number_input('Height (in cm)')
#Weight
p4=st.number_input('Weight')
#Duration
p5=st.number_input('Duration')
#Heart Rate
p6=st.number_input('Heart rate')
#Body Temperature
p7=st.number_input('Body Temperature(°C)')

#0 - Extremely Weak 1 - Weak 2 - Normal 3 - Overweight 4 - Obesity 5 - Extreme Obesity

if st.button('Predict'):
    query=np.array([p1,p2,p3,p4,p5,p6,p7],dtype=object).reshape(1,7)
    pred=Model.predict(query)
    st.title('THE CALORIES BURN{} in '.format(pred))

