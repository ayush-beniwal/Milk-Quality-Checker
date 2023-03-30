import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
import seaborn as sns
from PIL import Image
import random


modelinfo = {
    "Logistic Regression": "./models/LR.joblib",
    "Support Vector Machine": "./models/SVM.joblib",
    "K-Nearest Neighbor": "./models/KNN.joblib",
    "Decision Trees": "./models/DT.joblib",
    "Random Forest": "./models/rf.joblib",
    "Multi-layer Perceptron": "./models/mlpc.joblib",
    "Support Vector Machine (Tuned)": "./models/tuned_svm.joblib",
    "Gradient Boosting Classifier (Tuned)": "./models/tpot.joblib"
}
models = {}
for i, v in modelinfo.items():
    models[i] = joblib.load(open(v, "rb"))


def data_preprocessor(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.Taste = df.Taste.map({'Bad': 0, 'Good': 1})
    df.Odor = df.Odor.map({'Bad': 0, 'Good': 1})
    df.Fat = df.Fat.map({'Bad': 0, 'Good': 1})
    df.Turbidity = df.Turbidity.map({'Bad': 0, 'Good': 1})
    return df


def visualize_confidence_level(prediction_proba, model):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time
    return type : matplotlib bar chart
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(
        data=data, columns=['Percentage'])
    grad_percentage["Grade"] = ["Low", "Medium", "High"]

    fig, ax = plt.subplots()

    ax = sns.barplot(x=grad_percentage["Percentage"],
                     y=grad_percentage["Grade"], ax=ax, orient="h", palette="mako", order=["High", "Medium", "Low"])
    ax.bar_label(ax.containers[0])

    ax.set_xlim(xmin=0, xmax=100)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed',
                   alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Prediction Confidence Level (%)",
                  labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Milk Quality", labelpad=10, weight='bold', size=12)
    ax.set_title(model, fontdict=None,
                 loc='center', pad=None, weight='bold')

    pred = grad_percentage["Grade"][grad_percentage["Percentage"].idxmax()]
    st.write(f"I predict that you have **{pred}** grade milk")

    st.pyplot(fig)
    return


st.write("""
# Milk Quality Prediction ML Web-App 
This app predicts the **Quality of Milk**  using **milk features** input via the **side panel** 
""")

# user input parameter collection with streamlit side bar
st.sidebar.header('User Input Parameters')


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe
    """

    models = (
        "Logistic Regression",
        "Support Vector Machine",
        "K-Nearest Neighbor",
        "Decision Trees",
        "Random Forest",
        "Multi-layer Perceptron",
        "Support Vector Machine (Tuned)",
        "Gradient Boosting Classifier (Tuned)"
    )
    model = st.sidebar.selectbox("Model", models)

    pH = st.sidebar.slider('pH', min_value=6.0, value=6.6, max_value=7.0)
    Temperature = st.sidebar.slider(
        'Temperature (°C)', min_value=34.0, value=37.0, max_value=50.0)
    Colour = st.sidebar.slider(
        'Color (Pure white = 255)', min_value=240, value=255, max_value=255, step=1)
    Taste = st.sidebar.selectbox("Taste", ("Good", "Bad"))
    Odor = st.sidebar.selectbox("Odor", ("Good", "Bad"))
    Fat = st.sidebar.selectbox("Fat", ("Good", "Bad"))
    Turbidity = st.sidebar.selectbox("Turbidity", ("Good", "Bad"))

    features = {
        'pH': pH,
        'Temperature': Temperature,
        'Taste': Taste,
        'Odor': Odor,
        'Fat': Fat,
        'Turbidity': Turbidity,
        'Colour': Colour,
    }

    data = pd.DataFrame(features, index=[0])

    return data, model


user_input_df, model = get_user_input()
processed_user_input = data_preprocessor(user_input_df).copy()

st.subheader('User Input parameters')
user_input_df.replace({0: "Bad", 1: "Good"}, inplace=True)
st.write(user_input_df)


st.subheader('Prediction')
prediction_proba = models[model].predict_proba(processed_user_input)
visualize_confidence_level(prediction_proba, model)

model_df = {}
for parameter in models[model].get_params():
    v = models[model].get_params()[parameter]
    model_df[parameter] = v
model_df = pd.DataFrame([model_df])

st.subheader('Model Hyperparameters')
st.write(model_df)


st.subheader('Dataset')
st.write(
    "The dataset used can be found here: https://www.kaggle.com/datasets/cpluzshrijayan/milkquality")
