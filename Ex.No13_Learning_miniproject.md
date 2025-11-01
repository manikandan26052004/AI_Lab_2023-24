# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 25/10/2025                                                                     
### REGISTER NUMBER : 212222220022
### AIM: 
To write a program to train the classifier for Weather Condition Prediction using Decision Tree Classifier.
###  Algorithm:
Step 1: Start the program.

Step 2: Import the required libraries such as pandas, numpy, scikit-learn, and plotly.

Step 3: Load the weather dataset using pd.read_csv().

Step 4: Perform data preprocessing — handle missing values, encode categorical data, and normalize features.

Step 5: Split the dataset into training and testing sets using train_test_split().

Step 6: Train the model using the Decision Tree Classifier on the training data.

Step 7: Test the model on the testing data and generate predictions.

Step 8: Evaluate the model performance using metrics like accuracy, confusion matrix, and classification report.

### Program:
```py
!pip install scikit-learn==1.4.2 imbalanced-learn==0.12.2
!pip install catboost
import pandas as pd
import numpy as np
# Visualization
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

# Feature engineering
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Models (liner models)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# Models (tree-based models)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# training
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split


# Testing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


data = pd.read_csv('/seattle-weather.csv')
data

data.info()

data.describe()


data.isna().sum()


graph = px.line(data, x='date', y='precipitation', title='Seattle Weather Precipitation Over Time')
graph.update_layout(
    xaxis_title='Date',
    yaxis_title='Precipitation (inches)',
    title_x=0.5
)
```
### Output:
<img width="852" height="523" alt="Screenshot 2025-10-25 140421" src="https://github.com/user-attachments/assets/01a3c75d-0055-45ce-9aa5-25519e0c6f27" />
<img width="791" height="717" alt="image" src="https://github.com/user-attachments/assets/528b223d-ef87-4367-90d7-cf6a02c9915c" />
<img width="1441" height="608" alt="image" src="https://github.com/user-attachments/assets/4f1c8f1a-3b59-41a1-99e9-e0c6c097a140" />



### Result:
Thus the system was trained successfully and the prediction was carried out.
