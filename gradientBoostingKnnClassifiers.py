import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #plotting
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

path_to_file = 'Acoustic_Features.csv'
df = pd.read_csv(path_to_file)

df.rename(columns = {'Class':'Mood'}, inplace = True)

df["Mood"].value_counts().plot(
    kind="bar", color=["salmon", "lightblue", "lightgreen", "yellow"])
plt.xticks(rotation=0)

plt.style.use("default")
plt.style.use("seaborn-whitegrid")

y = df.iloc[:,0]
X = df.iloc[:,1:]

# split data into train and test sets
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

#GradientBoostingClassifier
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train, y_train)

# make predictions
y_pred_gb = model_gb.predict(X_test)

# evaluate predictions
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy: %.2f%%" % (accuracy_gb * 100.0))

#KNNClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# make predictions
y_pred_knn = knn.predict(X_test)

# evaluate predictions
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy: %.2f%%" % (accuracy_knn * 100.0))

# initialize list of lists
data = [['GradientBoostingClassifier', accuracy_gb], ['KNeighborsClassifier', accuracy_knn]]
  
# Create the pandas DataFrame
result = pd.DataFrame(data, columns=['Model', 'Accuracy'])
  
# print dataframe.
print(result)