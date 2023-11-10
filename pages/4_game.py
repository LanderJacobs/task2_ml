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

def button_press(id):
    
    st.session_state['game'][id] = st.session_state['current_symbol']
    predict()
    
    if check_win():
        if np.all(st.session_state['game'] != 'b'):
            st.session_state['win'] = 'x heeft verloren!'
        else: 
            st.session_state['win'] = 'x heeft gewonnnen!' if st.session_state['current_symbol'] == 'x' else 'x heeft verloren!'
    else:
        st.session_state['current_symbol'] = 'x' if st.session_state['current_symbol'] == 'o' else 'o'

def button_reset():
    st.session_state['current_symbol'] = 'x'
    st.session_state['game'] = np.array(['b','b','b','b','b','b','b','b','b'])
    st.session_state['win'] = ''
    st.session_state['f_pred'] = True
    st.session_state['n_pred'] = True
    st.session_state['b_pred'] = True

def check_win():
    if np.all(st.session_state['game'][0:3] == st.session_state['current_symbol']) or np.all(st.session_state['game'][3:6] == st.session_state['current_symbol']) or np.all(st.session_state['game'][6:] == st.session_state['current_symbol']):
        return True
    if np.all(st.session_state['game'][[0,3,6]] == st.session_state['current_symbol']) or np.all(st.session_state['game'][[1,4,7]] == st.session_state['current_symbol']) or np.all(st.session_state['game'][[2,5,8]] == st.session_state['current_symbol']):
        return True
    if np.all(st.session_state['game'][[0,4,8]] == st.session_state['current_symbol']) or np.all(st.session_state['game'][[2,4,6]] == st.session_state['current_symbol']):
        return True
    if np.all(st.session_state['game'] != 'b'):
        return True
    return False

def translate_values(value_array, number_array):
    value_list = list(value_array)
    number_list = list(number_array)
    return {x: y for x,y in zip(sorted(set(value_list), key=value_list.index), sorted(set(number_list), key=number_list.index))}

def predict():
    translated_game = np.array([int(st.session_state['ttt_data']['values'][str(y)][x]) for x, y in zip(st.session_state['game'], range(0,9))]).reshape(1,9)
    st.session_state['f_pred'] = list(st.session_state['forest']['model'].predict(translated_game))[0] == 'positive'
    st.session_state['n_pred'] = list(st.session_state['neighbour']['model'].predict(translated_game))[0] == 'positive'
    st.session_state['b_pred'] = list(st.session_state['bayes']['model'].predict(translated_game))[0] == 'positive'

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

if 'game' not in st.session_state:
    st.session_state['game'] = np.array(['b','b','b','b','b','b','b','b','b'])

if 'current_symbol' not in st.session_state:
    st.session_state['current_symbol'] = 'x'

if 'win' not in st.session_state:
    st.session_state['win'] = ''

if 'f_pred' not in st.session_state:
    st.session_state['f_pred'] = True

if 'n_pred' not in st.session_state:
    st.session_state['n_pred'] = True

if 'b_pred' not in st.session_state:
    st.session_state['b_pred'] = True

# app
st.title("Play Tic-Tac-Toe")
for x in range(0,7,3):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state['game'][0+x] == 'b' and st.session_state['win'] == '':
            st.button(st.session_state['current_symbol'], key="button"+str(0+x), on_click=button_press, args=[0+x])
        else:
            st.write('' if st.session_state['game'][0+x] == 'b' else st.session_state['game'][0+x])
    with col2:
        if st.session_state['game'][1+x] == 'b' and st.session_state['win'] == '':
            st.button(st.session_state['current_symbol'], key="button"+str(1+x), on_click=button_press, args=[1+x])
        else:
            st.write('' if st.session_state['game'][1+x] == 'b' else st.session_state['game'][1+x])

    with col3:
        if st.session_state['game'][2+x] == 'b' and st.session_state['win'] == '':
            st.button(st.session_state['current_symbol'], key="button"+str(2+x), on_click=button_press, args=[2+x])
        else:
            st.write('' if st.session_state['game'][2+x] == 'b' else st.session_state['game'][2+x])

st.button("Reset", on_click=button_reset)
st.subheader(st.session_state['win'])

with st.container():
    colf, coln, colb = st.columns(3)

    with colf:
        st.write("Het Random Forest denkt met " + str(round(st.session_state['forest']['accuracy'] * 100, 2)) + "% zekerheid:")
        st.write("x wint" if st.session_state["f_pred"] else "x verliest")
    with coln:
        st.write("De Neighbours denken met " + str(round(st.session_state['neighbour']['accuracy'] * 100, 2)) + "% zekerheid:")
        st.write("x wint" if st.session_state["n_pred"] else "x verliest")
    with colb:
        st.write("Bayes denkt met " + str(round(st.session_state['bayes']['accuracy'] * 100, 2)) + "% zekerheid:")
        st.write("x wint" if st.session_state["b_pred"] else "x verliest")
