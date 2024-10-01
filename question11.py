import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (replace with your actual dataset path)
df = pd.read_csv('try1.csv')

# Calculate the average value of the 'Total' column
average_transaction_value = df['Total'].mean()

# Create a new column 'Above_Average' where 1 = above average, 0 = below average
df['Above_Average'] = df['Total'].apply(lambda x: 1 if x >= average_transaction_value else 0)

# Select the relevant features
features = ['Product line', 'Quantity', 'Branch', 'Payment', 'Customer type']
X = df[features]  # Features
y = df['Above_Average']  # Target

# Perform one-hot encoding on the categorical columns
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shape of the training and test sets
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Initialize and train the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logreg = logreg_model.predict(X_test)

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

# Print the evaluation metrics for Logistic Regression
print("Logistic Regression:")
print(f"Accuracy: {accuracy_logreg:.2f}")
print(f"Precision: {precision_logreg:.2f}")
print(f"Recall: {recall_logreg:.2f}")
print(f"F1 Score: {f1_logreg:.2f}")

feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': logreg_model.coef_[0]})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance)
print("\n")

# Plot the feature importance for Logistic Regression model
importances = pd.Series(logreg_model.coef_[0], index=X_train.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title('Feature Importance')
plt.show()



accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# Print the evaluation metrics for Decision Tree
print("Decision Tree:")
print(f"Accuracy: {accuracy_dt:.2f}")
print(f"Precision: {precision_dt:.2f}")
print(f"Recall: {recall_dt:.2f}")
print(f"F1 Score: {f1_dt:.2f}")

feature_importance_dt = pd.DataFrame({'Feature': X.columns, 'Importance': dt_model.feature_importances_})
feature_importance_dt.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance_dt)


# Plot the feature importance for Decision Tree model
importances = pd.Series(dt_model.feature_importances_, index=X_train.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title('Feature Importance')
plt.show()

#Credit: Janshul Sharma, 
# for more details regarding this assignment, contact janshulsharma00@gmail.com