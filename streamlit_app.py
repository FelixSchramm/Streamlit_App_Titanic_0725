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
# This will be displayed on every page
st.sidebar.write("---")
st.sidebar.write("Made by Felix Schramm")
st.sidebar.write("Version: 1.0")
st.sidebar.write("Last updated: July 2025")

# --- Page 1: Exploration ---
if page == pages[0]: 
    st.write("### Presentation of data")
    st.write("This dataset contains information about the passengers on the Titanic, including whether they survived or not.")
    st.dataframe(df.head(10))
    st.write("Shape of the dataset:", df.shape)
    st.write("Descriptive statistics:")
    st.dataframe(df.describe())
    if st.checkbox("Show NA"):
        st.write("Sum of missing values per column:")
        st.dataframe(df.isna().sum())

# --- Page 2: Data Visualization ---
if page == pages[1]: 
    st.write("### DataVizualization")
    st.write("This section will provide visual insights into the dataset.")
    
    st.write("The countplot below shows the distribution of passengers who survived and those who did not.")
    fig = plt.figure()
    sns.countplot(x='Survived', data=df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x='Pclass', data=df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)
    
    fig = sns.displot(x='Age', data=df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    st.write("The following visualizations will show the relationship between survival and other features.")
    fig = plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    st.pyplot(fig)
    
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    st.write("Correlation Matrix Heatmap:")
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes(include=['number']).corr(), ax=ax, annot=True, cmap='coolwarm')
    st.pyplot(fig)

# --- Page 3: Modelling ---
if page == pages[2]: 
    st.write("### Modelling")

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
    classifier_name = st.selectbox("Select a classifier", ('Random Forest', 'SVC', 'Logistic Regression'))
    
    if st.button("Train Model"):
        # When button is clicked, train the model and store results in session_state
        clf = prediction(classifier_name, X_train, y_train)
        y_pred = clf.predict(X_test)
        
        st.session_state.accuracy = accuracy_score(y_test, y_pred)
        st.session_state.cm = confusion_matrix(y_test, y_pred)
        st.session_state.classifier_name = classifier_name
        st.session_state.model_trained = True

    # Check if a model has been trained in this session
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.write(f"### Results for: {st.session_state.classifier_name}")
        
        display_option = st.radio('What do you want to show?', ('Accuracy', 'Confusion Matrix'))

        if display_option == 'Accuracy':
            st.write(f"Accuracy: {st.session_state.accuracy:.2f}")
        elif display_option == 'Confusion Matrix':
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
