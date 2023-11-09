import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

def bayes():
    gnb = GaussianNB()

    gnb = gnb.fit(st.session_state.ttt_data["x_train"], st.session_state.ttt_data["y_train"])
    y_pred_g = gnb.predict(st.session_state.ttt_data["x_test"])
    return {"accuracy": accuracy_score(st.session_state.ttt_data["y_test"], y_pred_g), "confusion": confusion_matrix(st.session_state.ttt_data["y_test"], y_pred_g, labels=["positive", "negative"])}

# state variables
if 'depth' not in st.session_state:
    st.session_state['depth'] = 5

if "ttt_data" not in st.session_state:
    st.session_state["ttt_data"] = read_data()

if 'forest' not in st.session_state:
    st.session_state['forest'] = forest()

if 'naive' not in st.session_state:
    st.session_state['bayes'] = bayes()

# app
st.title("Naive Bayes")
st.write("Bayes laat niet echt toe om parameters aan te passen.")

st.write("Accuracy of Naive Bayes: " + str(round(st.session_state['bayes']['accuracy'] * 100, 2)) + " %")
st.subheader("Voorspellingen van Bayes:")
st.table(pd.DataFrame(st.session_state['bayes']['confusion'], columns=['x wins', 'o wins'], index=['x wins', 'o wins']))

st.write("Accuracy of Random Forest with depth of " + str(st.session_state['depth']) + ": " + str(round(st.session_state['forest']['accuracy'] * 100, 2)) + " %")
st.subheader("Voorspellingen van het random forest:")
st.table(pd.DataFrame(st.session_state['forest']['confusion'], columns=['x wins', 'o wins'], index=['x wins', 'o wins']))