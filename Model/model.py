# We first install and import the necessary libraries for our creating our ML model
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier

# Load the dataset from Kaggle
df = pd.read_csv('./diabetes.csv')
print(df.head())
print(df.info())

# Split the dataset into Training and Testing
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

# Re-scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Oversample minority class (non-diabetic, i.e., y==0)

# For instance, you might aim for balanced or some ratio like 50:50

# Get the counts of the majority class (y==0) and minority class (y==1)

counts = np.bincount(y_train)
majority_count = counts[0]
minority_count = counts[1]


# Set the sampling strategy to balance the classes
sampling_strategy = {0: majority_count, 1: majority_count}


sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)


x_res, y_res = sm.fit_resample(x_train, y_train)


print("Resampled counts:", np.bincount(y_res))

# Design the Pipeline for the ML Model
pipe = Pipeline([
    ('classifier', RandomForestClassifier())
])
param_grid = {
    'classifier': [RandomForestClassifier(), XGBClassifier(), LGBMClassifier(), LogisticRegression()],
    'classifier__n_estimators': [50, 100,]
}

# Applying Random forest Algorithm
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
bagging_rf = BaggingClassifier(
    estimator=rf_base,
    n_estimators=10,  # number of random forests
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
bagging_rf.fit(x_res, y_res)
y_pred_val = bagging_rf.predict(x_test)
print("Accuracy :", accuracy_score(y_test,y_pred_val.round()))

# Applying XGB Algorithm
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric= 'logloss'
)
model.fit(x_res, y_res)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred.round())
print("Accuracy:", accuracy)

# Applying LightGBM Algorithm
# To avoid the feature name warning, convert scaled data back to DataFrames
x_train_df = pd.DataFrame(x_res, columns=X.columns)
x_test_df = pd.DataFrame(x_res, columns=X.columns)
lgbm_model = LGBMClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)
lgbm_model.fit(x_train_df, y_train)
y_pred_val = lgbm_model.predict(x_test_df)
print("Accuracy :", accuracy_score(y_test,y_pred_val))

# Using LGR Algorithm
log_model = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
log_model.fit(x_res, y_res)

y_pred_val = log_model.predict(x_test)
print("Accuracy :", accuracy_score(y_test,y_pred_val))

# VotingClassifier to choose the Algorithm with best accuracy
voting_clf_soft = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('lb', LGBMClassifier()),
        ('xb',XGBClassifier())
        ],
    voting='soft',
)
voting_clf_soft.fit(x_res, y_res)
accuracy = voting_clf_soft.score(x_test, y_test)
print("Accuracy:", accuracy)

estimator = [
         ('lr',LogisticRegression(solver='liblinear', random_state=42)),
         ('rf',RandomForestClassifier(n_estimators=100,random_state=42)),
         ('lb',LGBMClassifier(n_estimators=100,random_state=42)),
         ('xb',XGBClassifier(n_estimators=100,random_state=42))
        ]

# Stacking (To combine multiple predictions from multiple base models)
# Done to improve the accuracy of the ML Model
Stacking_Clf= StackingClassifier(
    estimators=estimator,
    final_estimator=LogisticRegression(),
    passthrough=False,
    n_jobs=1
)
Stacking_Clf.fit(x_train,y_train)
y_pred_stack = Stacking_Clf.predict(x_test)
print("Accuracy :", accuracy_score(y_test,y_pred_stack))

# We have trained the ML Model
# Now we save it using Pickle, so we can pack our model as an API using FastAPI
with open('ml_model.pkl',"wb") as f:
    pickle.dump(model,f)