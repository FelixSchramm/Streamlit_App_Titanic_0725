import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# It's good practice to load the data once and cache it
@st.cache_data
def load_data():
    # This path needs to be relative to the script's location for deployment
    df = pd.read_csv("data/titanic/train.csv")
    return df

df = load_data()

st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

# --- Sidebar Info ---
st.sidebar.write("---")
st.sidebar.write("Made by Felix Schramm")
st.sidebar.write("Version: 1.0")
st.sidebar.write("Last updated: July 2025")

# --- Page 1: Exploration ---
if page == pages[0]: 
    st.header("1. Data Exploration")
    st.write("This section presents the raw data and provides a first look at the variables.")
    st.write("The goal is to predict whether a passenger survived the Titanic disaster or not. This is a binary classification problem.")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    st.write("**Shape of the dataset:**", df.shape)
    
    st.subheader("Descriptive Statistics")
    st.write("A summary of the numerical variables in the dataset.")
    st.dataframe(df.describe())
    
    if st.checkbox("Show Missing Values"):
        st.subheader("Sum of Missing Values")
        st.write("This shows the total count of missing values for each column.")
        st.dataframe(df.isna().sum())

# --- Page 2: Data Visualization ---
if page == pages[1]: 
    st.header("2. Data Visualization")
    st.write("This section provides visual insights into the dataset to understand the passenger demographics and their relationship with survival.")
    
    st.subheader("Distribution of Survival")
    st.write("This chart shows the number of passengers who survived (1) versus those who did not (0).")
    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)
    st.write("We can observe that the majority of passengers in this dataset did not survive. However, the classes are balanced enough for a classification model.")

    st.subheader("Passenger Demographics")
    st.write("The following charts describe the typical profile of a Titanic passenger.")
    
    fig = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Distribution of Passenger Gender")
    st.pyplot(fig)
    st.write("There were significantly more male passengers than female passengers.")

    fig = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Distribution of Passenger Class")
    st.pyplot(fig)
    st.write("The majority of passengers were traveling in the 3rd class.")
    
    fig = sns.displot(x='Age', data=df, kde=True)
    plt.title("Distribution of Passenger Age")
    st.pyplot(fig)
    st.write("The age of passengers is widely distributed, with a notable peak for young adults between 20 and 40 years old.")

    st.subheader("Survival Rate Analysis")
    st.write("Here, we analyze the impact of different factors on passenger survival.")

    fig = plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    plt.title("Survival Count by Gender")
    st.pyplot(fig)
    st.write("This chart clearly shows that females had a much higher survival rate than males.")

    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    plt.title("Survival Rate by Passenger Class")
    st.pyplot(fig)
    st.write("The survival rate decreases drastically with the class. First-class passengers had the highest chance of survival.")
    
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    plt.title("Survival Rate by Age and Class")
    st.pyplot(fig)
    st.write("Age and class both played a crucial role. Younger passengers (children) had a higher survival rate, especially those in 1st and 2nd class.")

    st.subheader("Correlation Matrix Heatmap")
    st.write("The heatmap shows the correlation between numerical variables. A value close to 1 or -1 indicates a strong correlation, while a value close to 0 indicates a weak one.")
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=['number']).corr(), ax=ax, annot=True, cmap='coolwarm')
    st.pyplot(fig)

# --- Page 3: Modelling ---
if page == pages[2]: 
    st.header("3. Modelling")
    st.write("In this section, we will train machine learning models to predict passenger survival based on the features.")

    # --- Data Preprocessing ---
    model_df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = model_df['Survived']
    X_cat = model_df[['Pclass', 'Sex',  'Embarked']]
    X_num = model_df[['Age', 'Fare', 'SibSp', 'Parch']]

    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())

    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # --- Model Training Function ---
    def prediction(classifier, X_train, y_train):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    # --- UI and State Management ---
    st.subheader("Model Selection and Training")
    classifier_name = st.selectbox("Select a classifier", ('Random Forest', 'SVC', 'Logistic Regression'))
    
    if st.button("Train Model"):
        clf = prediction(classifier_name, X_train, y_train)
        y_pred = clf.predict(X_test)
        
        st.session_state.accuracy = accuracy_score(y_test, y_pred)
        st.session_state.cm = confusion_matrix(y_test, y_pred)
        st.session_state.classifier_name = classifier_name
        st.session_state.model_trained = True

    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.subheader(f"Results for: {st.session_state.classifier_name}")
        
        display_option = st.radio('What do you want to show?', ('Accuracy', 'Confusion Matrix'))

        if display_option == 'Accuracy':
            st.write(f"Accuracy: {st.session_state.accuracy:.2f}")
        elif display_option == 'Confusion Matrix':
            st.write("The confusion matrix shows the performance of the classification model.")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
