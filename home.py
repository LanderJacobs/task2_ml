import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import streamlit as st

# methods
def read_data():
    tictactoe = pd.read_csv('resources/tic-tac-toe.data', sep=',', names=['top-left',  'top-middle', 'top-right', 'middle-left', 'middle-middle', 'right-middle', 'bottom-left', 'bottom-middle', 'bottom-right', 'win'])
    ttt_df = pd.DataFrame(tictactoe)
    feat_cols = ['top-left',  'top-middle', 'top-right', 'middle-left', 'middle-middle', 'right-middle', 'bottom-left', 'bottom-middle', 'bottom-right']
    x = ttt_df[feat_cols]
    y = ttt_df['win']

    ce_ord = ce.OrdinalEncoder(cols=feat_cols)
    x_cat = ce_ord.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_cat, y, test_size=0.3, random_state=0)
    
    # extra for game
    value_translation = {}

    for i in range(0,9):
        value_translation[str(i)] = translate_values(np.array(x)[:,i], np.array(x_cat)[:, i])
    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test, "values": value_translation}

def forest():
    rfc = RandomForestClassifier(criterion='entropy', max_depth=st.session_state.depth, n_estimators=st.session_state['amount'])

    rfc = rfc.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred = rfc.predict(st.session_state.ttt_data["x_test"])

    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred, labels=["positive", "negative"]), "model": rfc}

def neighbour():
    knc = KNeighborsClassifier(n_neighbors=st.session_state.neighbours)

    knc = knc.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred_n = knc.predict(st.session_state.ttt_data["x_test"])
    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred_n), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred_n, labels=["positive", "negative"]), "model": knc}

def bayes():
    gnb = GaussianNB()

    gnb = gnb.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred_g = gnb.predict(st.session_state.ttt_data["x_test"])
    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred_g), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred_g, labels=["positive", "negative"]), "model": gnb}

def translate_values(value_array, number_array):
    value_list = list(value_array)
    number_list = list(number_array)
    return {x: y for x,y in zip(sorted(set(value_list), key=value_list.index), sorted(set(number_list), key=number_list.index))}

# state variables
if 'depth' not in st.session_state:
    st.session_state['depth'] = 5

if 'amount' not in st.session_state:
    st.session_state['amount'] = 100
    
if 'neighbours' not in st.session_state:
    st.session_state['neighbours'] = 5

if "ttt_data" not in st.session_state:
    st.session_state["ttt_data"] = read_data()

if 'forest' not in st.session_state:
    st.session_state['forest'] = forest()

if 'neighbour' not in st.session_state:
    st.session_state['neighbour'] = neighbour()

if 'bayes' not in st.session_state:
    st.session_state['bayes'] = bayes()

# app
st.title("Benchmarking Machine Learning Algorithms")
st.subheader("van Lander Jacobs")
st.write("Ik heb voor deze taak een dataset met 958 mogelijke combinaties van Tic-Tac-Toe-spellen gekozen. In deze dataset zijn er 9 features, 1 voor elk mogelijk vakje, met de status van elk vakje: 'x', 'o' of 'b'(=blank). Ook is het interessant om te weten dat men er van uit gaat dat de x-speler het spel begint.")
st.write("Het is ook mogelijk om uit testen hoe accuraat de algoritmes kunnen berekenen of x wint door zelf een spelletje te spelen en te zien wat ze er van denken.")
st.write("Je kunt switchen tussen de pagina's en de parameters aanpassen zoals je wilt, maar als je de app refresht zul je zelfgekozen waarden opnieuw moeten ingeven.")
