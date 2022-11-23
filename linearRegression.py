import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Substitute the path_to_file content by the path to your student_scores.csv file 
path_to_file = 'student_scores.csv'
df = pd.read_csv(path_to_file)

df.plot.scatter(x='Hours', y='Scores', title='Scatterplot of hours and scores percentages')

print(df.corr())
print(df.describe())

# 2D input for LinearRegression() shape (25,) to (25,1)
y = df['Scores'].values.reshape(-1, 1)
X = df['Hours'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regressor.predict(X_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

# The Intercept
print("Intercept: \n", regressor.intercept_)

# The coefficients
print("Coefficients: \n", regressor.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()