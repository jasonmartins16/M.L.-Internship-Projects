import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv('E:\Lung Cancer\dataset_med.csv')
print(df.info())
print(df['survived'].value_counts())

# Convert date columns and create treatment_duration
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

#convert string data into numerical data
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['cancer_stage'] = df['cancer_stage'].map({'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3 })
df['smoking_status'] = df['smoking_status'].map({'Never Smoked': 0, 'Former Smoker': 1, 'Passive Smoker': 2, 'Current Smoker': 3})
df['treatment_type'] = df['treatment_type'].map({'Chemotherapy': 0, 'Surgery': 1, 'Radiation': 2, 'Combined': 3})

#handle outliers
df['bmi'] = df['bmi'].clip(lower=10, upper=50)
df['cholesterol_level'] = df['cholesterol_level'].clip(lower=100, upper=350)

# Encode binary categorical columns
binary_cols = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'family_history']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Drop unused columns
df.drop(['id', 'diagnosis_date', 'end_treatment_date', 'country'], axis=1, inplace=True)

# Define features and label
X = df.drop('survived', axis=1)
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale all numerical features (excluding encoded one-hot columns)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Check for missing values in the dataset
print(df.isnull().sum())

#Train the model(random-forrest)
model_rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
model_rfc.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_rfc.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the full pipeline
joblib.dump(model_rfc, 'predict_survival.pkl')
