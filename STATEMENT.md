# Aryan Amit Arya 
# 25BAI10295
# VITyarthi Assignment

#  COVID-19 Detection using Machine Learning

## Project Title

**COVID-19 Detection using Machine Learning: A Comparative Analysis of Classification Algorithms**

-----

##  Overview of the Project

This project addresses the critical need for **swift and reliable diagnosis** of the highly contagious **COVID-19** virus. We harness the power of **Machine Learning (ML)** to develop a system for early detection. The core methodology involves:

1.  Analyzing readily available COVID-19 data.
2.  Comparing the **accuracies** of various classification models (K-Nearest Neighbors, Random Forest, Naive Bayes).
3.  Selecting the **best-performing algorithm** to build the final predictive model.

The goal is to create an efficient system that predicts a person's COVID-19 status based on input features, thereby assisting in preventative measures and resource allocation.

-----

##  Features

The developed system includes the following key features:

  * **Comparative Analysis:** Provides a quantitative comparison of the predictive accuracy of multiple ML models (KNN, Random Forest, Naive Bayes).
  * **Predictive Model:** Utilizes the most accurate algorithm to predict the presence of the COVID-19 virus in an individual.

<div>
  <img src="https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcRuOY5n-zSutY3TxQC8pcbH0l8HRSjkWFSBQcI4Cobt7blIlWb4-67QSqYTUBPFcInkM0wYukMKIqSO1yGPLj-M-zum8caRccB5MOMDv9A6TWbHgcA"
</div>

  * **Data-Driven Diagnosis:** Offers a non-invasive, rapid screening tool based on readily available data (e.g., patient symptoms or laboratory results, depending on the dataset used).
  * **Model Evaluation:** Reports standard classification metrics (e.g., Accuracy, Precision, Recall, F1-Score) to assess the model's performance.

-----

##  Technologies/Tools Used

This project relies on standard data science and machine learning libraries in a Python environment:

  * **Programming Language:** **Python 3.x**
  * **Core Libraries:** **NumPy** (for numerical operations) and **Pandas** (for data manipulation and analysis).
  * **Machine Learning Framework:** **Scikit-learn (sklearn)** (for implementing KNN, Random Forest, Naive Bayes, model training, and evaluation).
  * **Visualization:** **Matplotlib** and **Seaborn** (for data exploration and results plotting).
  * **Environment Management:** Recommended use of **Conda** or **venv** (Python virtual environment).

-----

##  Steps to Install & Run the Project

Follow these steps to set up the project environment and run the main script:

### 1\. Clone the Repository

```bash
git clone [repository_link_here]
cd [project_folder_name]
```

### 2\. Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
conda create -n covid-env python=3.9
conda activate covid-env

python -m venv covid-env
source covid-env/bin/activate  # On Linux/macOS
.\covid-env\Scripts\activate   # On Windows
```

### 3\. Install Required Dependencies

Install all necessary packages using the `requirements.txt` file (if provided):

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not available, manually install the main dependencies:*

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 4\. Run the Main Script

Execute the primary Python script to perform the data loading, model training, comparison, and testing:

```bash
python main_detection_script.py
```

*(Note: Replace `main_detection_script.py` with the actual name of your primary execution file.)*

-----

## Instructions for Testing

The final predictive system can be tested to verify its performance and generalization ability.

### 1\. Automatic Testing

The main script automatically performs cross-validation and splits the data into training and testing sets.

  * **Check the Console Output:** After running the script, the console will display the **accuracy scores** for KNN, Random Forest, and Naive Bayes, followed by the final performance metrics (e.g., **Confusion Matrix**, Classification Report) of the selected best model on the unseen **test data**.

### 2\. Manual Prediction Testing

To test the model with a new, single data point:

1.  Open the project's Jupyter Notebook or Python prediction script.
2.  Define a new input vector (a list or array of features) matching the structure used during training:
    ```python
    new_patient_features = [[Fever_Yes=1, DryCough_Yes=1, Fatigue_Yes=1, ...]] 
    ```
3.  Load the saved, trained model (e.g., using `pickle` or `joblib`):
    ```python
    import joblib
    model = joblib.load('best_model.pkl')
    ```
4.  Use the model's `predict()` method on the new data:
    ```python
    prediction = model.predict(new_patient_features)
    print(f"Prediction: {'COVID-19 Positive' if prediction[0] == 1 else 'COVID-19 Negative'}")
    ```
