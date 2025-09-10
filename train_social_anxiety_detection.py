# train_social_anxiety_detection.py

#import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

#load data
df= pd.read_csv("social_anxiety.csv")

# Separate features and target
X = df.drop(['Anxiety Level (1-10)'], axis=1)
y = df['Anxiety Level (1-10)']

# i split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# split tinto num_cols and cat_cols
num_cols= X.select_dtypes(include=np.number).columns.tolist()
cat_cols= X.select_dtypes(include= 'object').columns.tolist()

#column transformer
num_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy= 'median')),
    ('scaler', StandardScaler())
])
cat_transformer= Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(sparse_output=False, drop='first',handle_unknown='ignore'))
])

#combine our transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

#define the best model
model= LinearRegression()

pipeline= Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])
pipeline.fit(X_train, y_train)

#save our model
with open('social_anxiety.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
    
print("Model trained and saved as social_anxiety.pkl")

#Detect on an example
data= {
       'Age': [20],
       'Age_cat': ['Adult'],
       'Gender': ['Female'],
       'Occupation': ['Data Scientist'],
       'Sleep Hours': [5.0],
       'Physical Activity (hrs/week)': [2.0],
       'Caffeine Intake (mg/day)': [0],
       'Alcohol Consumption (drinks/week)': [0],
       'Smoking': ['No'],
       'Family History of Anxiety': ['No'],
       'Stress Level (1-10)': [9],
       'Heart Rate (bpm)': [80],
       'Breathing Rate (breaths/min)': [20],
       'Sweating Level (1-5)': [3],
       'Dizziness': ['No'],
       'Medication': ['No'],
       'Therapy Sessions (per month)': [0],
       'Recent Major Life Event': ['No'],
       'Diet Quality (1-10)': [5]
}

#convert to a dataframe
sample_df= pd.DataFrame(data)

#load the model
with open('social_anxiety.pkl', 'rb') as f:
    model= pickle.load(f)
    
#predict
detection= model.predict(sample_df)


print(f"The Level of Anxiety is {detection}")











 