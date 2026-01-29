from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib

#Load dataset
df = pd.read_csv('E:\Heart Disease\dataset.csv')
X = df.drop("target", axis = 1) # features
y = df["target"] # 0 = no disease, 1 = disease

#Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2)

#Scale the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model selection
#model = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight = 'balanced')
model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=5, class_weight='balanced')
model.fit(X_train_scaled, y_train)

#Predict outcomes
#y_pred = model.predict(X_test_scaled)
prob = model.predict_proba(X_test_scaled)[0][1]
print("Probability:", prob)

prob = model.predict_proba(X_test_scaled)
pred_class = (prob[:, 1] >= 0.4).astype(int)

#Evaluate (optional)
print(classification_report(y_test, pred_class))
print(confusion_matrix(y_test, pred_class))

#Save Model
joblib.dump(model, "heart_disease_predictor.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")