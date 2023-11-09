import pandas as pd

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

    x_train, x_test, y_train, y_test = train_test_split(x_cat, y, test_size=0.3)
    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

def forest():
    rfc = RandomForestClassifier(criterion='entropy', max_depth=st.session_state.depth, n_estimators=100)

    rfc = rfc.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred = rfc.predict(st.session_state.ttt_data["x_test"])

    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred, labels=["positive", "negative"])}

def neighbour():
    knc = KNeighborsClassifier(n_neighbors=st.session_state.neighbours)

    knc = knc.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred_n = knc.predict(st.session_state.ttt_data["x_test"])
    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred_n), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred_n, labels=["positive", "negative"])}

def bayes():
    gnb = GaussianNB()

    gnb = gnb.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred_g = gnb.predict(st.session_state.ttt_data["x_test"])
    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred_g), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred_g, labels=["positive", "negative"])}

# state variables
if 'depth' not in st.session_state:
    st.session_state['depth'] = 5

if 'neighbours' not in st.session_state:
    st.session_state['neighbours'] = 5

if "ttt_data" not in st.session_state:
    st.session_state["ttt_data"] = read_data()

if 'forest' not in st.session_state:
    st.session_state['forest'] = forest()

if 'neighbour' not in st.session_state:
    st.session_state['neighbour'] = neighbour()

if 'naive' not in st.session_state:
    st.session_state['bayes'] = bayes()

# app
st.title("Benchmarking Machine Learning Algorithms")
st.subheader("van Lander Jacobs")
st.write("Je kunt switchen tussen de pagina's en de states zullen behouden worden, maar als je de pagina refresht zul je zelfgekozen waarden opnieuw moeten ingeven.")
st.write("Ik heb voor deze taak een dataset met mogelijke combinaties van Tic-Tac-Toe-spellen gekozen.")
st.write("Het is belangrijk om te begrijpen dat voor elk van de spellen men er vanuit gaat dat de x-speler begint.")