import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot,plot
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)

bodyfat = pd.read_csv('https://raw.githubusercontent.com/josephkhoury98/Healthcare-Analytics/main/bodyfat.csv')
bodyfat.info()
print(bodyfat.head())

countries_weight = pd.read_csv ('https://raw.githubusercontent.com/josephkhoury98/Healthcare-Analytics/main/Countries_fat.csv', encoding = 'latin-1')
#print(countries_weight.head())
countries_weight.info()
st.set_page_config(layout="wide")
st.markdown(f"<h1 style='text-align:center;' >{'<b>Body Fat Application</b>'}</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align:center;' >{'by Joseph El Khoury'}</h3>", unsafe_allow_html=True)
st.write("")
st.markdown(f"<h4 style='text-align:center;' >{'Choose a Page'}</h4>", unsafe_allow_html=True)
col7,col8,col9= st.beta_columns(3)

page = col8.selectbox("", ['About','Overview of Obesity Around the World','Body Fat & Prediction'])

#Prepare dataset for prediction to save time on page loading
pred_data = bodyfat.copy()
X = pred_data.drop('BodyFat', axis=1)
y = pred_data['BodyFat'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

def predict_bf (Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist):
    prediction = regr.predict([[Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist]])
    return prediction




if page == 'About':
    st.subheader('Welcome to the body fat dashboard!')
    st.write('Thank you for stopping by! In this dashboard, we will be covering the evolution of obesity in the world throughout the years. We will also take a look at the different factors affecting the body fat percentage, as well as predicting your own body fat percentage.')
    st.write("")
    st.write("")
    st.subheader('Sources:')
    st.write('[] The dataset was generously supplied by Dr. A. Garth Fisher who gave permission to freely distribute the data and use for non-commercial purposes: https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset')
    st.write('[] Another dataset was also extracted from the NCD RisC website, a website managed by a network of health scientists around the world that provides rigorous and timely data on major risk factors for non-communicable diseases for all of the world’s countries: https://www.ncdrisc.org/data-downloads-adiposity.html')
elif page == 'Overview of Obesity Around the World':
    html_title_sec1= """
            <div style="background-color:steelblue;padding:0px">
            <h2 style="color:white;text-align:center;"> Obesity Around the World </h2>
            </div>
            """
    html_title_sec2= """
            <div style="background-color:steelblue;padding:0px">
            <h2 style="color:white;text-align:center;"> Obesity Over the Years </h2>
            </div>
            """
    ######################################################################################################
    ##############################Obesity in the World##################################################

    st.markdown(html_title_sec1, unsafe_allow_html=True)
    col1, col2 = st.beta_columns((1,2))
    #creating map for Obesity prevalence
    col1.write("")
    col1.write("")
    col1.write("")
    col1.write("")
    col1.markdown(f"<h4 style='text-align:left; font-family:arial;' >{'Slide through the years to see the progress of obesity in the world'}</h4>", unsafe_allow_html=True)
    year = col1.slider('Year', min_value=1975, max_value=2016)
    sex = col1.selectbox("Sex", ['Men','Women'])
    df= countries_weight[countries_weight['Year'] == year]
    df = df[df['Sex'] == sex]
    data = dict(
            type = 'choropleth',
            colorscale = 'RdBu',
            reversescale=True,
            locations = df['Country'],
            locationmode = "country names",
            z = df['Prevalence of BMI>=30 kg/m² (obesity)'],
            text = df['Country'],
            colorbar = {'title' : 'Prevalence of BMI>=30 kg/m²','titleside':'right'},
            )
    layout = dict(geo = dict(projection = {'type':'equirectangular'}))
    choromap = go.Figure(data = [data],layout = layout)

    iplot(choromap,validate=False)
    choromap.update_layout(width=1000, margin={"r":0,"t":0,"l":100,"b":0})
    col2.plotly_chart(choromap)

    ######################################################################################################
    ##############################Obesity over the years##################################################

    st.markdown(html_title_sec2, unsafe_allow_html=True)
    st.write("")
    col3, col4, col5, col6, col7 = st.beta_columns([30,1,30,5,30])

    col3.write('')
    col3.write('')
    col3.write('')
    col3.write('')

    country = col3.selectbox("Country", ["Afghanistan","Albania","Algeria","American Samoa","Andorra","Angola","Antigua and Barbuda","Argentina","Armenia","Australia",
    "Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin","Bermuda","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana",
    "Brazil","Brunei Darussalam","Bulgaria","Burkina Faso","Burundi","Cabo Verde","Cambodia","Cameroon","Canada","Central African Republic","Chad","Chile","China","China (Hong Kong SAR)",
    "Colombia","Comoros","Congo","Cook Islands","Costa Rica","Cote d'Ivoire","Croatia","Cuba","Cyprus","Czech Republic","Denmark","Djibouti","Dominica","Dominican Republic","DR Congo","Ecuador",
    "Egypt","El Salvador","Equatorial Guinea","Eritrea","Estonia","Ethiopia","Fiji","Finland","France","French Polynesia","Gabon","Gambia","Georgia","Germany","Ghana","Greece","Greenland","Grenada",
    "Guatemala","Guinea","Guinea Bissau","Guyana","Haiti","Honduras","Hungary","Iceland","India","Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan","Kenya",
    "Kiribati","Kuwait","Kyrgyzstan","Lao PDR","Latvia","Lebanon","Lesotho","Liberia","Libya","Lithuania","Luxembourg","Macedonia (TFYR)","Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands",
    "Mauritania","Mauritius","Mexico","Micronesia (Federated States of)","Moldova","Mongolia","Montenegro","Morocco","Mozambique","Myanmar","Namibia","Nauru","Nepal","Netherlands","New Zealand","Nicaragua",
    "Niger","Nigeria","Niue","North Korea","Norway","Occupied Palestinian Territory","Oman","Pakistan","Palau","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal","Puerto Rico",
    "Qatar","Romania","Russian Federation","Rwanda","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa","Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles",
    "Sierra Leone","Singapore","Slovakia","Slovenia","Solomon Islands","Somalia","South Africa","South Korea","Spain","Sri Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syrian Arab Republic",
    'Taiwan',"Tajikistan","Tanzania","Thailand","Timor-Leste","Togo","Tokelau","Tonga","Trinidad and Tobago","Tunisia","Turkey","Turkmenistan","Tuvalu","Uganda","Ukraine","United Arab Emirates","United Kingdom",
    "United States of America","Uruguay","Uzbekistan","Vanuatu","Venezuela","Viet Nam","Yemen","Zambia","Zimbabwe"])
    sex_country = col3.selectbox("Sex of interest", ['Men','Women'])

    countries_fat = countries_weight[countries_weight['Country'] == country]
    df2 = countries_fat[countries_fat['Sex'] == sex_country]
    mean_bmi_chart = px.line(df2, x=df2["Year"], y=df2["Mean BMI"],color_discrete_sequence=['steelblue'])
    mean_bmi_chart.update_layout(width = 450, margin={"r":0,"t":0,"l":0,"b":0})
    #mean_bmi_chart.update_layout(xaxis={'visible': False, 'showticklabels': False})
    mean_bmi_chart.update_layout(template="simple_white")
    col5.markdown(f"<h3 style='text-align:right; font-family:arial;' >{'Average BMI Throughout the Years'}</h3>", unsafe_allow_html=True)
    col5.plotly_chart(mean_bmi_chart)

    #####################################################col6.image('vertical_line.png')

    years2 = col3.slider('Year of interest (for Pie Chart only)', min_value=1975, max_value=2016)
    df_prevalence = df2[df2['Year']==years2]
    df_final=df_prevalence.copy()
    df_final['Non-obese'] = 1 - df_prevalence['Prevalence of BMI>=30 kg/m² (obesity)']
    data_imp=df_final[['Prevalence of BMI>=30 kg/m² (obesity)','Non-obese']].values.tolist()
    data=data_imp[0]
    labels = ['Obese','Non-obese']
    prevalence_obese_chart= px.pie(data, values= data, names=labels,color_discrete_sequence=["steelblue","firebrick"])
    prevalence_obese_chart.update_layout(width = 450, margin={"r":0,"t":0,"l":30,"b":0})
    prevalence_obese_chart.update_layout(template="simple_white")
    col7.markdown(f"<h3 style='text-align:center; font-family:arial;' >{'Percentage of obese vs non-obese'}</h3>", unsafe_allow_html=True)
    col7.plotly_chart(prevalence_obese_chart)


    ######################################################################################################
    ######################################Last Page#######################################################
elif page == 'Body Fat & Prediction':
    html_title_sec3= """
            <div style="background-color:steelblue;padding:0px">
            <h2 style="color:white;text-align:center;"> Body fat and Body Mass Index </h2>
            </div>
            """
    html_title_sec4= """
            <div style="background-color:steelblue;padding:0px">
            <h2 style="color:white;text-align:center;"> Predictions of Body Fat Percentage </h2>
            </div>
            """

    st.markdown(html_title_sec3, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.beta_columns([30,1,30,2,30])
    bodyfat_weight_chart = px.scatter(bodyfat, x=bodyfat["Weight"], y=bodyfat["BodyFat"], trendline = 'ols', color_discrete_sequence=["steelblue"])
    bodyfat_weight_chart.update_layout(width = 450, margin={"r":0,"t":50,"l":0,"b":0})
    bodyfat_weight_chart.update_layout(template="simple_white", title = 'Body Fat Percentage vs Weight')

    bodyfat_height_chart = px.scatter(bodyfat, x=bodyfat["Height"], y=bodyfat["BodyFat"], trendline = 'lowess', color_discrete_sequence=["steelblue"])
    bodyfat_height_chart.update_layout(  width = 450, margin={"r":0,"t":50,"l":0,"b":0})
    bodyfat_height_chart.update_layout(template="simple_white", title = 'Body Fat Percentage vs Height')

    col1.write('')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.write('')
    col1.markdown(f"<h4 style='text-align:left; font-family:arial;' >{'Choose whether you want to visualize Bodyfat vs Weight or vs Height'}</h4>", unsafe_allow_html=True)


    chart = col1.selectbox('',['Weight','Height'])

    if chart == "Weight":
        col3.plotly_chart(bodyfat_weight_chart)
    else:
        col3.plotly_chart(bodyfat_height_chart)
    #   col4.image('vertical_line.png')

    weight_height_chart = px.scatter(bodyfat, x=bodyfat["Height"], y=bodyfat["Weight"], trendline = 'lowess', color_discrete_sequence=["steelblue"])
    weight_height_chart.update_layout(width = 450, margin={"r":0,"t":50,"l":0,"b":0})
    weight_height_chart.update_layout(template="simple_white", title = 'Weight vs Height')
    col5.plotly_chart(weight_height_chart)



    st.markdown(html_title_sec4, unsafe_allow_html=True)
    #Density, Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Writst
    col6,col7,col8 = st.beta_columns(3)
    Density_input = col7.text_input('Density',"1.072")
    Age_input = col6.slider('Age', min_value=0, max_value=120)
    Weight_input = col8.text_input('Weight in Kgs','83')
    Height_input = col6.text_input('Height in cm','190')
    Neck_input = col7.text_input("Neck Circumference in cm",'38')
    Chest_input =col8.text_input("Chest Circumference in cm",'96')
    Abdomen_input = col6.text_input("Abdomen Circumference in cm",'90')
    Hip_input =col7.text_input("Hip Circumference in cm",'105')
    Thigh_input =col8.text_input("Thigh Circumference in cm",'63')
    Knee_input =col6.text_input("Knee Circumference in cm",'40')
    Ankle_input =col7.text_input("Ankle Circumference in cm",'23')
    Biceps_input =col8.text_input("Biceps Circumference in cm",'35')
    Forearm_input =col6.text_input("Forearm Circumference in cm",'27')
    Wrist_input =col7.text_input("Wrist Circumference in cm",'18')

    Density_input_float = float(Density_input)
    Age_input_float = float(Age_input)
    Weight_input_float = float(Weight_input)
    Height_input_float = float(Height_input)
    Neck_input_float =float(Neck_input)
    Chest_input_float =float(Chest_input)
    Abdomen_input_float = float(Abdomen_input)
    Hip_input_float =float(Hip_input)
    Thigh_input_float =float(Thigh_input)
    Knee_input_float =float(Knee_input)
    Ankle_input_float =float(Ankle_input)
    Biceps_input_float =float(Biceps_input)
    Forearm_input_float =float(Forearm_input)
    Wrist_input_float =float(Wrist_input)


    BMI = Weight_input_float/Height_input_float/Height_input_float*10000

    if col6.button('Predict!'):
        bodyfat_percentage = predict_bf (Density_input_float, Age_input_float, Weight_input_float, Height_input_float, Neck_input_float,
        Chest_input_float, Abdomen_input_float, Hip_input_float, Thigh_input_float, Knee_input_float, Ankle_input_float, Biceps_input_float, Forearm_input_float, Wrist_input_float)
        st.success('With an accuracy of 80%, your body fat percentage is: {}%'.format(bodyfat_percentage))
        st.success('Your BMI is {}'.format(BMI))

footer= """<style>
a:link , a:visited{
color: black;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: teal;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/joseph-el-khoury1998/" target="_blank">Joseph El Khoury</a></p>

</div>
"""
st.markdown(footer,unsafe_allow_html=True)
