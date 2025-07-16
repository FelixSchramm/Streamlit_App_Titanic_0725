# Titanic Survival Prediction - Streamlit App

This project is an interactive web application built with Streamlit to analyze and predict the survival of passengers on the Titanic.

## Features

The application is divided into three main sections:

1.  **Exploration:**
    * Displays the raw data and basic information such as dataset shape and statistical summaries.
    * Provides an option to show the sum of missing values per column.

2.  **Data Visualization:**
    * Visualizations for the distribution of demographic features (Gender, Class, Age).
    * Plots showing the relationship between various features and the survival rate.
    * A correlation heatmap to display the relationships between numerical variables.

3.  **Modelling:**
    * Choose between different classification models (Random Forest, SVC, Logistic Regression).
    * Train the selected model with the click of a button.
    * View the results as either accuracy or a confusion matrix.

## How to Run the App

1.  **Clone the repository:**
    ```bash
    git clone <git@github.com:FelixSchramm/Streamlit_App_Titanic_0725.git>
    cd <your-repository-folder>
    ```

2.  **Install dependencies:**
    Make sure you have a `requirements.txt` file with the following content, then run the command:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```
