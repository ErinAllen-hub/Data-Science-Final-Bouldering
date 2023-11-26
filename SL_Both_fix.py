#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import streamlit as st

df = pd.read_csv('Combined_graphs.csv')


# In[7]:


st.title('Bouldering Ability Dashboard')
st.write('Here is a dashboard showing the effects of various strength and body composition features on a persons ability to boulder.')


# In[8]:


def plot_gender():
    st.write('This pie chart shows the gender distribution of boulderers in the dataset.')
    grouped = df.groupby(df.columns[0]).count().max(axis=1)

    labels = ['Female', 'Male', 'Other']
    colors = ['purple', 'thistle', 'indigo']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(grouped, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title('Gender of Boulderers')
    ax.set_ylabel('')

    ax.legend(labels=labels)

    st.pyplot(fig)

plot_gender()


# In[9]:


st.write('Select a graph of the various features related to climbing to see the distributions')

def plot_height():
    fig, ax = plt.subplots()
    ax.hist(df[df.columns[5]], bins=int(df[df.columns[5]].max()-df[df.columns[5]].min()),
            color='purple', edgecolor='white', linewidth=1.2)
    ax.set_title('Height Distribution')
    ax.set_xlabel('Height (cm)')
    ax.set_ylabel('Count')
    return fig

def plot_Weight():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.hist(df[df.columns[6]], bins=int(df[df.columns[6]].max()-df[df.columns[6]].min()),
            color='purple', edgecolor='white', linewidth=1.2)
    ax.set_title('Weight Distribution')
    ax.set_xlabel('Weight (KG)')
    ax.set_ylabel('Count')
    return fig

def plot_Span():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.hist(df[df.columns[7]], bins=int(df[df.columns[7]].max()-df[df.columns[7]].min()),
            color='purple', edgecolor='white', linewidth=1.2)
    ax.set_title('Span Distribution')
    ax.set_xlabel('Span (cm)')
    ax.set_ylabel('Count')
    return fig
    
def plot_maxb():
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.hist(df[df.columns[1]], bins=int(df[df.columns[1]].max()-df[df.columns[1]].min()),
            color='purple', edgecolor='white', linewidth=1.2)
    ax.set_title('Max Boulder Distribution')
    ax.set_xlabel('Max Boulder')
    ax.set_ylabel('Count')
    return fig
    
plot_choice = st.selectbox('Select a plot', ("Height Distribution", "Weight Distribution","Span Distribution","Max Boulder Distribution" ))

if plot_choice == "Height Distribution":
    fig = plot_height()
    st.pyplot(fig)
elif plot_choice == "Weight Distribution":
    fig = plot_Weight()
    st.pyplot(fig)
elif plot_choice == "Span Distribution":
    fig = plot_Span()
    st.pyplot(fig)
elif plot_choice == "Max Boulder Distribution":
    fig = plot_maxb()
    st.pyplot(fig)


# In[10]:


st.write('This shows the comparison of male and female climbers on the various included features')

def plot_gender_profile():
    profile_vars = ['exp', 'pullup', 'pushup', 'height (cm)', 'weight (kg)', 'span (cm)', 'BMI', 'APE', 'pullup_ratio']

    fig, axs = plt.subplots(ncols=2, figsize=(20,5))

    male_data = df.loc[df['gender'] == 'Male', profile_vars]
    male_data.mean().plot.bar(title="Male climber profile", ax=axs[0], color='purple')

    female_data = df.loc[df['gender'] == 'Female', profile_vars]
    female_data.mean().plot.bar(title="Female climber profile", ax=axs[1], color='purple')

    axs[0].set_xlabel('Variables')
    axs[0].set_ylabel('Mean Value')
    axs[1].set_xlabel('Variables')
    axs[1].set_ylabel('Mean Value')
    fig.suptitle('Male vs Female Climber Profile', fontsize=20)

    return fig

fig = plot_gender_profile()
st.pyplot(fig)


# In[11]:


st.write('The below graph demonstrates how important each of the selected features is for climbing ability and maximum boulder a climber can do')

def spearman(frame, features):
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['spearman'] = [frame[f].corr(frame['max_boulder'], 'spearman') for f in features]
    spr = spr.sort_values('spearman')

    palette = sns.color_palette("Purples_r", len(spr))

    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='spearman', orient='h', palette=palette)
    plt.title('Spearman correlation with max_boulder')
    plt.xlabel('Spearman correlation')
    
quant_feat =['sex', 'exp', 'pullup', 'pushup', 'height (cm)', 'weight (kg)', 'span (cm)', 'BMI', 'APE', 'pullup_ratio']

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(spearman(df, quant_feat))


# In[15]:


def plot_climbing_metrics(df):
    compare_vars = ["exp", "pullup", "pushup", "height (cm)", "weight (kg)", "span (cm)", "BMI", "APE", "pullup_ratio"]

    grouped = df.groupby("top_or_bottom")[compare_vars].mean()

    diff_mean = grouped.loc["Top"] - grouped.loc["Bottom"]

    fig, ax = plt.subplots(figsize=(10,6))
    diff_mean.plot.barh(ax=ax, color="purple")

    ax.set_title("Comparison of Top and Bottom Climbers on Selected Metrics")
    ax.set_xlabel("Mean Difference")

    st.pyplot(fig)

    st.write('This graph shows the mean difference between the top and bottom climbers for each selected metric. A positive value means that the mean of the top climbers is higher than the mean of the bottom climbers, and a negative value means that the mean of the bottom climbers is higher. Looking at the graph, we can see that the top climbers have higher values for pullup, pushup, APE, and pullup_ratio, while the bottom climbers have higher values for BMI. The other metrics do not show a clear difference between the top and bottom climbers. Therefore, we can say that the top climbers generally have better upper body strength (as indicated by higher values for pullup and pushup), longer arms (as indicated by higher values for APE), and a lower BMI, while the bottom climbers have a higher BMI.')
    
plot_climbing_metrics(df)


# In[ ]:


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
    bmi = weight / ((height*100)**2)
    ape = arm_span / height 
    pullup_ratio = (weight + max_pullups)/weight

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




