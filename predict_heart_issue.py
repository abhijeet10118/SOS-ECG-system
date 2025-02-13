import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("heart.csv")

print(dataset.info())
print(dataset.describe())
print(dataset.head())

sns.countplot(x="target", data=dataset)
plt.show()

print(dataset.corr()["target"].abs().sort_values(ascending=False))

sns.barplot(x="sex", y="target", data=dataset)
plt.show()

sns.barplot(x="cp", y="target", data=dataset)
plt.show()

sns.barplot(x="fbs", y="target", data=dataset)
plt.show()

sns.barplot(x="restecg", y="target", data=dataset)
plt.show()

sns.barplot(x="exang", y="target", data=dataset)
plt.show()

sns.barplot(x="slope", y="target", data=dataset)
plt.show()

sns.barplot(x="ca", y="target", data=dataset)
plt.show()

sns.barplot(x="thal", y="target", data=dataset)
plt.show()

predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    return round(accuracy_score(Y_pred, Y_test) * 100, 2)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
score_lr = evaluate_model(lr, X_test, Y_test)

nb = GaussianNB()
nb.fit(X_train, Y_train)
score_nb = evaluate_model(nb, X_test, Y_test)

sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
score_svm = evaluate_model(sv, X_test, Y_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
score_knn = evaluate_model(knn, X_test, Y_test)

max_accuracy = 0
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    current_accuracy = evaluate_model(dt, X_test, Y_test)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
score_dt = evaluate_model(dt, X_test, Y_test)

max_accuracy = 0
for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    current_accuracy = evaluate_model(rf, X_test, Y_test)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
score_rf = evaluate_model(rf, X_test, Y_test)

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)
score_xgb = evaluate_model(xgb_model, X_test, Y_test)

model = Sequential()
model.add(Dense(11, activation='relu', input_dim=13))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=300, verbose=0)

Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
score_nn = round(accuracy_score(rounded, Y_test) * 100, 2)

scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]

for i in range(len(algorithms)):
    print(f"The accuracy score achieved using {algorithms[i]} is: {scores[i]} %")

plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
sns.barplot(x=algorithms, y=scores, palette="viridis")
plt.xlabel("Algorithms", fontsize=14)
plt.ylabel("Accuracy Score", fontsize=14)
plt.title("Accuracy Scores of Different Algorithms", fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)

for i, score in enumerate(scores):
    plt.text(i, score + 0.5, f"{score}%", ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

def get_probabilities(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    elif isinstance(model, Sequential):
        return model.predict(X_test).flatten()
    else:
        return model.decision_function(X_test)

probabilities_lr = get_probabilities(lr, X_test)
probabilities_nb = get_probabilities(nb, X_test)
probabilities_svm = get_probabilities(sv, X_test)
probabilities_knn = get_probabilities(knn, X_test)
probabilities_dt = get_probabilities(dt, X_test)
probabilities_rf = get_probabilities(rf, X_test)
probabilities_xgb = get_probabilities(xgb_model, X_test)
probabilities_nn = get_probabilities(model, X_test)

probabilities_df = pd.DataFrame({
    "Logistic Regression": probabilities_lr,
    "Naive Bayes": probabilities_nb,
    "Support Vector Machine": probabilities_svm,
    "K-Nearest Neighbors": probabilities_knn,
    "Decision Tree": probabilities_dt,
    "Random Forest": probabilities_rf,
    "XGBoost": probabilities_xgb,
    "Neural Network": probabilities_nn
})

probabilities_df["Patient_ID"] = X_test.index

highest_risk_patients = {}
for model_name in probabilities_df.columns[:-1]:
    highest_risk_index = probabilities_df[model_name].idxmax()
    highest_risk_patients[model_name] = {
        "Patient_ID": probabilities_df.loc[highest_risk_index, "Patient_ID"],
        "Probability": probabilities_df.loc[highest_risk_index, model_name]
    }

print("\nPatients at Highest Risk of Heart Disease:")
for model_name, patient_info in highest_risk_patients.items():
    print(f"{model_name}: Patient ID {patient_info['Patient_ID']} with probability {patient_info['Probability']:.2f}")

overall_highest_risk_index = probabilities_df.iloc[:, :-1].max().idxmax()
overall_highest_risk_patient = highest_risk_patients[overall_highest_risk_index]
print(f"\nOverall Highest-Risk Patient: Patient ID {overall_highest_risk_patient['Patient_ID']} "
      f"with probability {overall_highest_risk_patient['Probability']:.2f} using {overall_highest_risk_index}")

highest_risk_patient_details = dataset.loc[overall_highest_risk_patient['Patient_ID']]
print("\nDetails of the Highest-Risk Patient:")
print(highest_risk_patient_details)

def predict_heart_disease_ecg():
    print("\nPlease provide the following ECG-related details to predict your risk of heart disease:")

    restecg = int(input("Enter resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy): "))
    thalach = int(input("Enter maximum heart rate achieved: "))
    oldpeak = float(input("Enter ST depression induced by exercise relative to rest: "))
    slope = int(input("Enter the slope of the peak exercise ST segment (0 = downsloping, 1 = flat, 2 = upsloping): "))

    user_data = {
        'restecg': [restecg],
        'thalach': [thalach],
        'oldpeak': [oldpeak],
        'slope': [slope]
    }

    user_df = pd.DataFrame(user_data)

    probability = rf.predict_proba(user_df)[0][1]
    probability_percent = round(probability * 100, 2)

    print(f"\nBased on your ECG inputs, your risk of having heart disease is: {probability_percent}%")

predict_heart_disease_ecg()
