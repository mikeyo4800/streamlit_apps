import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

with open("model.pkl", "rb") as f:
    model = joblib.load(f)


st.title("NBA Draft Classification")
st.subheader("Here we're using sklearn's GradientBoostingClassifier to predict a player's NBA draft round based on their college stats.")
st.subheader("All the college stats used in this application to predict whether a college player should be drafted in round 1 or round 2 shown significant difference in their means across both groups.")
st.write("For source code/github repo: [GitHub](https://github.com/mikeyo4800/draft_prospect_ml)")


rec_rank = st.slider('Recruiter Rank:', 0, 100)
fig, ax = plt.subplots()
ax.set_title('Recruiter Rank')
ax.bar([0,1, 2], [rec_rank, 81.29792387543262, 77.58388059701495], color='g')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)




dbpm = st.slider('dbpm:', -20.0, 20.0, value=1.0)
fig, ax = plt.subplots()
ax.set_title('Defense Box Plus Minus')
ax.bar([0,1, 2], [dbpm, 2.550139311988043, 2.0196380823960896], color='r')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)


obpm = st.slider('obpm: ', -20.0, 20.0, value=1.0)
fig, ax = plt.subplots()
ax.set_title('Offense Box Plus Minus')
ax.bar([0,1, 2], [obpm, 3.1359924562929775, 2.536770340464551], color='b')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)


bpm = st.slider('bpm: ', -20.0, 20.0, value=1.0)
fig, ax = plt.subplots()
ax.set_title('Box Plus Minus')
ax.bar([0,1, 2], [bpm, 5.686131497159942, 4.556407999511004], color='g')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)


stops = st.slider('stops: ', 0.0, 20.0)
fig, ax = plt.subplots()
ax.set_title('Total Stops')
ax.bar([0,1, 2], [stops, 175.11860453662192, 166.30285481540324], color='r')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)

dporpag = st.slider('dporpag:', -10.0, 10.0, value=1.0)
fig, ax = plt.subplots()
ax.set_title('Defense Players over Replacement')
ax.bar([0,1, 2], [dporpag, 3.287882841255603, 3.112296583374084], color='b')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)

porpag = st.slider('porpag:', -10.0, 10.0, value=1.0)
fig, ax = plt.subplots()
ax.set_title('Players over Replacement')
ax.bar([0,1, 2], [porpag, 3.244555286547085, 2.9539587642542786], color='g')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)


ftr = st.slider('ftr: ', 0.0, 100.0)
fig, ax = plt.subplots()
ax.set_title('Free Throw Rate')
ax.bar([0,1, 2], [ftr, 41.86905829596411, 38.79119804400977], color='r')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)

stl_per = st.slider('stl_per: ', 0.0, 5.0)
fig, ax = plt.subplots()
ax.set_title('Steals Per Game')
ax.bar([0,1, 2], [stl_per, 2.106576980568015, 1.9518337408312973], color='y')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)

FTA = st.slider('FTA: ', 0.0, 150.0)
fig, ax = plt.subplots()
ax.set_title('Total Free Throw Attempts')
ax.bar([0,1, 2], [FTA, 119.31539611360239, 109.34474327628362], color='b')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)

ORB_per = st.slider('ORB_per: ', 0.0, 15.0)
fig, ax = plt.subplots()
ax.set_title('Offensive Rebounds Per Game')
ax.bar([0,1, 2], [ORB_per, 6.81958146487294, 6.326772616136918], color='g')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['User Input', 'Round 1 Players', 'Round 2 Players'])
st.pyplot(fig)



X = {'Rec Rank': [rec_rank], 'dbpm': [dbpm], 'obpm': [obpm], 'bpm': [bpm], 'stops': [stops], 'dporpag': [dporpag], 'porpag': [porpag], 'ftr': [ftr], 'stl_per': [stl_per], 'FTA': [FTA], 'ORB_per': [ORB_per], 'yr': ['So'], 'AFFILIATION': ['Kanas'], 'conf': ['B12']}

X_data = pd.DataFrame(data=X)



pred = model.predict(X_data)[0]

if pred == 1:
    answer = 'Round 1'
else:
    answer = 'Round 2'

st.subheader("Prediction: {} ".format(answer))
