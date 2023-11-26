#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import joblib
import streamlit as st


# In[69]:


df = pd.read_csv("GH_data_2.csv")


# In[71]:


# Select features and target variable
features = ['Sex', 'Height (cm)', 'Weight (KG)', 'Arm Span (cm)',
       'How long have you been climbing for (years)?',
       'Frequency of climbing sessions per week',
       'Average hours climbing per week (not including training)',
       'Average hours Training for climbing per week ',
       'Campus Board frequency per week ',
       'Campus Board time per week (hours)',
       'Frequency of Endurance training sesions per week',
       'General Strength Training frequency per week ',
       'Time spent General strength training per week (hours)',
       'Max pull up reps', 'Max push ups reps', 'BMI', 'APE', 'Pullup ratio']
target = 'Hardest V Grade ever climbed '


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Fit linear regression model to training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model on test data
mse = np.mean((model.predict(X_test) - y_test) ** 2)
r_squared = model.score(X_test, y_test)

print("Mean squared error: ", mse)
print("R-squared: ", r_squared)


# In[72]:


joblib.dump(model, 'model.pkl')


# In[73]:


model = joblib.load('model.pkl')

def predict_v_grade(data):
    prediction = model.predict(data)
    return prediction


# In[76]:


def app():
    st.title('Climbing Grade Predictor')
    st.write('Enter the following details to predict your climbing grade')

    sex_mapping = {'Male': 1, 'Female': 2}
    sex = st.selectbox('Sex', list(sex_mapping.keys()), format_func=lambda x: x)
    sex = sex_mapping[sex]
    height = st.number_input('Height (cm)')
    weight = st.number_input('Weight (KG)')
    arm_span = st.number_input('Arm Span (cm)')
    climbing_experience = st.number_input('How long have you been climbing for (years)?')
    frequency_sessions = st.number_input('Frequency of climbing sessions per week')
    avg_hours_climbing = st.number_input('Average hours climbing per week (not including training)')
    avg_hours_training = st.number_input('Average hours Training for climbing per week')
    campus_board_freq = st.number_input('Campus Board frequency per week')
    campus_board_time = st.number_input('Campus Board time per week (hours)')
    endurance_freq = st.number_input('Frequency of Endurance training sesions per week')
    strength_freq = st.number_input('General Strength Training frequency per week')
    strength_time = st.number_input('Time spent General strength training per week (hours)')
    max_pullups = st.number_input('Max pull up reps')
    max_pushups = st.number_input('Max push ups reps')
    bmi = st.number_input('BMI')
    ape = st.number_input('APE')
    pullup_ratio = st.number_input('Pullup ratio')

    data = pd.DataFrame({'Sex': sex, 'Height (cm)': height, 'Weight (KG)': weight, 'Arm Span (cm)': arm_span,
            'How long have you been climbing for?': climbing_experience,
            'Frequency of climbing sessions per week': frequency_sessions,
            'Average hours climbing per week (not including training)': avg_hours_climbing,
            'Average hours Training for climbing per week ': avg_hours_training,
            'Campus Board frequency per week ': campus_board_freq,
            'Campus Board time per week (hours)': campus_board_time,
            'Frequency of Endurance training sesions per week': endurance_freq,
            'General Strength Training frequency per week ': strength_freq,
            'Time spent General strength training (hours)': strength_time, 'Max pull up reps': max_pullups,
            'max push ups reps': max_pushups, 'BMI': bmi, 'APE': ape, 'pullup_ratio': pullup_ratio}, index=[0])

    if st.button('Predict'):
        prediction = predict_v_grade(data)
        prediction_mapped = None
        for key, value in mapping.items():
            if key[0] <= prediction[0] <= key[1]:
                prediction_mapped = value
                break
        st.write(f'Your predicted hardest V grade is: {prediction_mapped}')

mapping = {
        (0.0, 0.99): 'V1-V2',
        (1.0, 2.00): 'V3',
        (2.1, 3.0): 'V4',
        (3.1, 4.0): 'V5',
        (4.1, 5.0): 'V6',
        (5.1, 6.0): 'V7',
        (6.1, 7.0): 'V8',
        (7.1, 8.0): 'V9',
        (8.1, 9.0): 'V10',
        (9.1, 10.0): 'V11',
        (10.1, 11.0): 'V12',
        (11.1, 12.0): 'V13',
        (12.1, 13.0): 'V14',
        (13.1, 14.0): 'V15',
        (14.1, 15.0): 'V16',
        (15.1, 16.0): 'V17',
        (16.1, 17.0): 'V18',
    }


# In[77]:


if __name__ == '__main__':
    app()


# In[ ]:




