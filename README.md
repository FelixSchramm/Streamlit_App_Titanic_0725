# Titanic Survival Prediction

### Problem Statement
The goal of this project is to build a machine learning model to predict who survived the Titanic disaster. The application guides users through the process of data exploration, model training, and evaluation for this classic classification problem.

### Data Source
The data for this project comes from the **"Titanic: Machine Learning from Disaster"** competition on Kaggle. The dataset consists of two main files: `train.csv`, used for training the model, and `test.csv`, used for making predictions.

### Tech Stack
* **Python**: The core programming language used for the project.
* **Streamlit**: Used to build the interactive web application.
* **Pandas & NumPy**: For data manipulation and numerical operations.
* **Scikit-learn**: The machine learning library used to build the Logistic Regression model.
* **Matplotlib & Seaborn**: For data visualization and plotting the confusion matrix.

### Key Findings
The project successfully developed a classification model using **Logistic Regression**. The app's final model achieved an accuracy of **77.27%** on the test dataset. The application provides a visual confusion matrix and a detailed classification report to help understand the model's performance on different classes (survived vs. not survived).

### Deployed App
https://titanic-predication-model.streamlit.app/ 

### How to Run
1.  Ensure you have Python installed.
2.  Install the required libraries from the `requirements.txt` file using the following command:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit application from your terminal with this command:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  The app will open automatically in your web browser, allowing you to interact with the survival prediction model.
