import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

#Load the data
df = pd.read_csv('E:\Mobile Phone Pricing\dataset.csv')

#Set the name of target column
target = 'price_range'

#Separate features and target
X = df.drop(target, axis= 1)
y = df[target]

#train-test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#train the xgb model
model = xgb.XGBClassifier(
    objective = 'multi:softmax',
    num_class = len(np.unique(y)),
    eval_metric = 'mlogloss',
    #use_label_encoder = False, No need if label_encoder is'nt included
    n_estimators = 100,
    max_depth = 6,
    learning_rate = 0.1,
)

model.fit(X_train, y_train)

#evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Training columns:", X.columns)

#save the model
joblib.dump(model, 'xgb_model.pkl')