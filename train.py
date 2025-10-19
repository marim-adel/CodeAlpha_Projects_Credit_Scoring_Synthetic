import numpy as np, pandas as pd, joblib, os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
# create synthetic credit dataset
X, y = make_classification(n_samples=2000, n_features=8, n_informative=5, weights=[0.7,0.3], random_state=42)
columns = ['age','income','existing_debt','credit_amount','credit_history_len','num_loans','num_defaults','employment_years']
df = pd.DataFrame(X, columns=columns)
# make some columns positive-like
df['income'] = (df['income'] - df['income'].min())*10000 + 20000
df['existing_debt'] = (df['existing_debt'] - df['existing_debt'].min())*1000
df['credit_amount'] = (df['credit_amount'] - df['credit_amount'].min())*5000
df['target'] = y
os.makedirs('data', exist_ok=True)
df.to_csv('data/credit_synthetic.csv', index=False)
X = df.drop('target', axis=1); y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, clf.predict_proba(X_test_s)[:,1]))
joblib.dump({'model':clf, 'scaler':scaler, 'columns': list(X.columns)}, 'models/credit_model.joblib')
print('Saved model and data to models/credit_model.joblib and data/credit_synthetic.csv')
