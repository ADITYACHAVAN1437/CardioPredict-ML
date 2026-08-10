import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("heart.csv")

df.head()

df.shape

df.info()
df.describe()
df.duplicated().sum()

df["HeartDisease"].value_counts().plot(kind="bar")

df.isnull().sum()


def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)


plotting("Age", 1)
plotting("RestingBP", 2)
plotting("Cholesterol", 3)
plotting("MaxHR", 4)


plt.tight_layout()


df["Cholesterol"].value_counts()

ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()

df["Cholesterol"] = df["Cholesterol"].replace(0, ch_mean)
df["Cholesterol"] = df["Cholesterol"].round(2)

resting_bp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()

df["RestingBP"] = df["RestingBP"].replace(0, resting_bp_mean)

df["RestingBP"] = df["RestingBP"].round(2)


def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)


plotting("Age", 1)
plotting("RestingBP", 2)
plotting("Cholesterol", 3)
plotting("MaxHR", 4)


plt.tight_layout()


sns.countplot(x=df["Sex"], hue=df["HeartDisease"])

sns.countplot(x=df["ChestPainType"], hue=df["HeartDisease"])

sns.countplot(x=df["FastingBS"], hue=df["HeartDisease"])

sns.boxplot(x="HeartDisease", y="Cholesterol", data=df)

sns.violinplot(x="HeartDisease", y="Age", data=df)

sns.heatmap(df.corr(numeric_only=True), annot=True)


df_encode = pd.get_dummies(df, drop_first=True)
df_encode

df_encode = df_encode.astype(int)
df_encode

from sklearn.preprocessing import StandardScaler

numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
df_encode.head()

df_encode.columns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X = df_encode.drop("HeartDisease", axis=1)
y = df_encode["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True),
}
results = []
for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": round(acc, 4), "F1 Score": round(f1, 4)})

results

import joblib
joblib.dump(models["KNN"],'KNN_heart.pkl')
joblib.dump(scaler,'Scaler.pkl')
joblib.dump(X.columns.tolist(),'columns.pkl')