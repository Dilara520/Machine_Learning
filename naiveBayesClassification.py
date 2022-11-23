import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('Naive-Bayes-Classification-Data.csv')

feature_cols = ['glucose', 'bloodpressure']
X = df[feature_cols] # Features
y = df.diabetes # Target variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

y_pred = model_nb.predict(X_test)

accuracy_nb = accuracy_score(y_test,y_pred)*100