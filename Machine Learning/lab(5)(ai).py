"""Task No 1:
Follow the instructions provided in lab file to apply Decision Tree using Heart attack prediction dataset."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
df = pd.read_csv("heart_attack_prediction_dataset.csv")
print(df.head())
df.rename(columns={"Sex": "Gender"}, inplace=True)
df = df.drop(columns=["Patient ID","Country","Continent","Hemisphere","Income"])
feature_cols = ["Age","Gender","Cholesterol","Heart Rate","Diabetes","Family History","Smoking","Obesity","Alcohol Consumption","Exercise Hours Per Week","Diet","Previous Heart Problems","Medication Use","Stress Level","Sedentary Hours Per Day","BMI","Triglycerides","Physical Activity Days Per Week","Sleep Hours Per Day"]
X = df[feature_cols]
y = df["Heart Attack Risk"]
X = pd.get_dummies(X, columns=["Gender", "Diet"])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
clf = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=1)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test,y_pred,pos_label=1))
print("Recall:", recall_score(y_test,y_pred,pos_label=1))



"""Task No 2:
Follow the instructions provided in lab file to apply Decision Tree using Churn Dataset."""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("telecom_churn.csv")
print(df.head())
feature_cols = ["AccountWeeks","ContractRenewal","DataPlan","DataUsage","CustServCalls","DayMins","DayCalls","MonthlyCharge","OverageFee","RoamMins"]
X = df[feature_cols]
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
dt = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test,dt_pred,pos_label=1)
dt_recall = recall_score(y_test,dt_pred,pos_label=1)
dt_f1 = f1_score(y_test,dt_pred,pos_label=1)
rf = RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test,rf_pred,pos_label=1)
rf_recall = recall_score(y_test,rf_pred,pos_label=1)
rf_f1 = f1_score(y_test,rf_pred,pos_label=1)
comparison = pd.DataFrame({"Metric": ["Accuracy","Precision","Recall","F1-Score"],"Decision Tree": [dt_accuracy,dt_precision,dt_recall,dt_f1],"Random Forest": [rf_accuracy,rf_precision,rf_recall,rf_f1]})
print("\nComparative Analysis:")
print(comparison)
dt_average = (dt_accuracy +dt_precision +dt_recall +dt_f1) / 4
rf_average = (rf_accuracy +rf_precision +rf_recall +rf_f1) / 4
print("\nAverage Performance:")
print("Decision Tree:", dt_average)
print("Random Forest:", rf_average)
if rf_average > dt_average:
    print("\nRandom Forest performed better than Decision Tree")
elif dt_average > rf_average:
    print("\nDecision Tree performed better than Random Forest")
else:
    print("\nBoth algorithms performed equally")



"""Task No 3:
Use Random Forest supervised learning algorithms on the same dataset. Create a comparative analysis table of their evaluation metrics, and clearly explain all steps, indicating which algorithm outperformed the other. Use all the applicable evaluation metrics."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv("telecom_churn.csv")
print(df.head())
feature_cols = ["AccountWeeks","ContractRenewal","DataPlan","DataUsage","CustServCalls","DayMins","DayCalls","MonthlyCharge","OverageFee","RoamMins"]
X = df[feature_cols]
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
dt = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, pos_label=1)
dt_recall = recall_score(y_test, dt_pred, pos_label=1)
dt_f1 = f1_score(y_test, dt_pred, pos_label=1)
rf = RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, pos_label=1)
rf_recall = recall_score(y_test, rf_pred, pos_label=1)
rf_f1 = f1_score(y_test, rf_pred, pos_label=1)
lr = LogisticRegression(max_iter=1000,random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, pos_label=1)
lr_recall = recall_score(y_test, lr_pred, pos_label=1)
lr_f1 = f1_score(y_test, lr_pred, pos_label=1)
comparison = pd.DataFrame({"Metric": ["Accuracy","Precision","Recall","F1-Score"],"Decision Tree": [dt_accuracy,dt_precision,dt_recall,dt_f1],"Random Forest": [rf_accuracy,rf_precision,rf_recall,rf_f1],"Logistic Regression": [lr_accuracy,lr_precision,lr_recall,lr_f1]})
print("\nComparative Analysis:")
print(comparison)
dt_average = (dt_accuracy +dt_precision +dt_recall +dt_f1) / 4
rf_average = (rf_accuracy +rf_precision +rf_recall +rf_f1) / 4
lr_average = (lr_accuracy +lr_precision +lr_recall +lr_f1) / 4
print("\nAverage Performance:")
print("Decision Tree:", dt_average)
print("Random Forest:", rf_average)
print("Logistic Regression:", lr_average)
results = {"Decision Tree": dt_average,"Random Forest": rf_average,"Logistic Regression": lr_average}
best_algorithm = max(results, key=results.get)
print("\nBest Performing Algorithm:", best_algorithm)