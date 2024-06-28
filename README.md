
# CUSTOMER CHURN PREDICTION APP

Cypha Tech is a leading telecommunications company that provides a comprehensive range of communication services, including phone, internet, and television, to both residential and business customers. In a highly dynamic and competitive market, the company strives to maintain its edge by offering innovative services and exceptional customer experiences.

To further enhance customer retention, the marketing department aims to proactively identify customers who are likely to churn. By predicting the likelihood of customer churn, Cypha Tech can implement targeted retention strategies, thereby ensuring sustained customer satisfaction and loyalty.

## Data Source

Data was sourced from [Kaggle](https://https://www.kaggle.com/datasets/rashadrmammadov/customer-churn-dataset)

The dataset contains information about customers and their churn status. Each row represents a customer, and each column contains customer attributes and information.

Data was already clean upon download

## Column Descriptions

```txt
**customerID:** Unique identifier for each customer.
**gender:** Gender of the customer (Male, Female).
**SeniorCitizen:** Whether the customer is a senior citizen or not (1: Yes, 0: No).
**Partner:** Whether the customer has a partner or not (Yes, No).
**Dependents:** Whether the customer has dependents or not (Yes, No).
**Tenure:** Number of months the customer has stayed with the company.
**PhoneService:** Whether the customer has a phone service or not (Yes, No).
**MultipleLines:** Whether the customer has multiple lines or not (Yes, No, No phone service).
**InternetService:** Type of internet service the customer has (DSL, Fiber optic, No).
**OnlineSecurity:** Whether the customer has online security or not (Yes, No, No internet service).
**OnlineBackup:** Whether the customer has online backup or not (Yes, No, No internet service).
**DeviceProtection:** Whether the customer has device protection or not (Yes, No, No internet service).
**TechSupport:** Whether the customer has tech support or not (Yes, No, No internet service).
**StreamingTV:** Whether the customer has streaming TV or not (Yes, No, No internet service).
**StreamingMovies:** Whether the customer has streaming movies or not (Yes, No, No internet service).
**Contract:** The contract term of the customer (Month-to-month, One year, Two year).
**PaperlessBilling:** Whether the customer has paperless billing or not (Yes, No).
**PaymentMethod:** The payment method of the customer (Electronic check, Mailed check, Bank transfer, Credit card).
**MonthlyCharges:** The amount charged to the customer monthly.
**TotalCharges:** The total amount charged to the customer.
**Churn:** Whether the customer churned or not (Yes, No).
```

# Machine Learning Model

```Language: Python```
```Model: LogisticRegression```

``` python
# Import required libraries
import numpy as np
import pandas as pd 
# For cross-validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# For fitting the model
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
# For model evaluation
from sklearn.metrics import roc_auc_score

# Save the model
import pickle
```

## Data Preparation

``` python
# Read data into a dataframe
df = pd.read_csv('../data/Telco_Customer_Churn.csv')

# Change column names into lower case and replace space between them with underscore ('_')
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Get column names of all categorical variables
categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)

# Change data in categorical variables to lower case and replace space between them with underscore ('_')
for col in categorical_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# Convert data in totalcharges to numeric
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')

# Handle missing values in totalcharges
df.totalcharges = df.totalcharges.fillna(df.totalcharges.mean())

# Convert churn variable to 0(No) and 1(Yes)
df.churn = (df.churn == 'yes').astype(int)
```

### Train Test Split

```This splits the data into a training and testing sets```

``` python
# Split the data into 20% for testing and 80% for training and testing.
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Split the training data further into 80% for training and 20% for validation.
# To getting 20% of the entire data, we'll need 25% of the remaining training data
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

#      60%          20%          20%
len(df_train), len(df_val), len(df_test)
```

``` python
# Get the target values for the training, validation, and test sets.
y_train = df_train.churn.values
y_valid = df_val.churn.values
y_test = df_test.churn.values

# Delete Target variable from the train dataset
del df_train['churn']
del df_test['churn']
del df_val['churn']
```

``` python
numerical = ['monthlycharges', 'tenure', 'totalcharges']

categorical = ['gender',
               'seniorcitizen', 
               'partner', 
               'dependents', 
               'phoneservice', 
               'multiplelines', 
               'internetservice', 
               'onlinesecurity', 
               'onlinebackup', 
               'deviceprotection', 
               'techsupport', 
               'streamingtv', 
               'streamingmovies', 
               'contract', 
               'paperlessbilling', 
               'paymentmethod']
```

## Training the Model

```python
dv = DictVectorizer(sparse=False)

# Transform datasets in to dictionaries
train_dict = df_train[categorical + numerical].to_dict(orient='records')
val_dict = df_val[categorical + numerical].to_dict(orient='records')

# Fit and transform the training data
X_train = dv.fit_transform(train_dict)
# Only transform the validation and test data
X_valid = dv.transform(val_dict)

# Define the model and fit or train the model 
model = LogisticRegression()
model.fit(X_train, y_train)
```

## Evaluation: Cross Validation

```The KFold class from sklearn.model_selection is used to split the data into n_splits folds, ensuring that each fold is used once as a validation set while the remaining folds form the training set. This process helps in assessing the model's robustness and its ability to generalize to unseen data.```

``` python
# Define the number of splits for cross-validation and the regularization parameter
n_split = 5
C = 1.0

# Initialize K-Fold with number of splits, shuffling, and a random state for reproducibility
kfold = KFold(n_splits=n_split, shuffle=True, random_state=1)

# Initialize a list to store AUC scores
scores = []

# Loop through each fold
for train_index, validation_index in kfold.split(df_train_full):
    # Split the data into training and validation sets based on the current fold
    df_train = df_train_full.iloc[train_index]
    df_validation = df_train_full.iloc[validation_index]

    # Extract the target variable for training and validation sets
    y_train = df_train.churn
    y_val = df_validation.churn

    # Train the model on the training data
    dv, model = train(df_train, y_train, C=C)
    
    # Predict the target variable on the validation data
    y_pred = predict(dv, df_validation, model)

    # Calculate the AUC score for the current fold
    auc = roc_auc_score(y_val, y_pred)
    
    # Append the AUC score to the list
    scores.append(auc
                  
# Print the mean and standard deviation of the AUC scores
print('C=%s %.3f +- %.3f'%(C, np.mean(scores), np.std(scores)))
```

``` txt
**Output**
C=1.0 0.842 +- 0.007
```

### Create functions for training and pridictions

```python
def train(df_train, y_train, C=1.0):
# Convert the data into a dictionary
    dicts = df_train[categorical + numerical].to_dict(orient='records')
# DictVectorizer is used for OneHotEndoding the categorical variables
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
# Model
    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model 
```

```python
def predict(dv, df, model):
    # Converts the data into a dictionary
    dicts = df[categorical + numerical].to_dict(orient='records')
   # Transform the data for prediction
    X = dv.transform(dicts)
   # Predict the probability  
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred
```

### Auc Score

``` python
dv, model = train(df_train_full, df_train_full.churn.values, C=1.0)
y_pred = predict(dv, df_test, model)
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc
```

``` txt
**Output**
0.8583084164797903
```

## Save modle with Pickle

```python
output_file = f'model_C={C}.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
```

# Deploy Model

## Building Web App with Flask

Ensure your directory looks like this

```bash
/my-flask-app/
├── app.py
├── model_C=1.0.bin
├── requirements.txt
└── templates/
    └── index.html
```

### requirements.txt file

```bash
Flask==2.0.3
Werkzeug==2.0.3
gunicorn==20.1.0
scikit-learn==1.0.2
numpy==1.21.4
```

### app.py

```python
from flask import Flask, request, jsonify, render_template
import pickle
## Load the model
input_file = 'model_C=1.0.bin'
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # customer = request.get_json()
        customer = request.form.to_dict()

        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[0, 1]
        churn = y_pred >= 0.5  # Assuming a threshold of 0.5 for churn prediction
        result = {
            'churn_probability': round(float(y_pred),2),
            'churn': bool(churn)
        }

        # Render the same template with the prediction results
        return render_template('index.html', prediction_results = result)
    # For GET request, render the template without results    
    return render_template('index.html')
   
if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(host='0.0.0.0', port=8080)
```

Your `index.html` file should contain your HTML code to render your UI
It must contain a form to submit the required data to send to the flask app for prediction.

To run this application on your local machine,

- Create a virtual environment

```bash
py m venv venv
```

- Activate your environment variable

```bash
source venv/Scripts/activate
```

- `cd` into the root directory
- Install your dependencies in the `requirements.txt` file

```bash
pip install -r requirements.txt
```

- Run the `app.py` file

```bash
py app.py
```

- Output

```bash
 * Serving Flask app 'churn'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. 
Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://192.168.88.143:8080
Press CTRL+C to quit
```

### Web App

![image](/Screenshot%202024-06-28%20124034.jpg)
