import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_lifestyle(df):
    # Drop irrelevant columns
    df = df.drop(columns=['PatientID', 'DoctorInCharge'])

    # Separate features and target
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    # Encode categorical variables
    categorical_cols = ['Gender', 'Ethnicity', 'EducationLevel']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Scale numeric features
    numeric_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
                    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
                    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                    'MMSE', 'FunctionalAssessment', 'ADL']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler

if __name__ == "__main__":
    df = pd.read_csv("../data/alzheimers_disease_data.csv")
    X, y, scaler = preprocess_lifestyle(df)
    print("Preprocessed Features Shape:", X.shape)
    print("Target Shape:", y.shape)
# Copilot helped with categorical encoding and scaling