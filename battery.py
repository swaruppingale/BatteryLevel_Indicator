# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# # Load dataset
# data = pd.read_csv('dummyDataset.csv')

# # Remove '%' sign and convert 'Battery %' column to numeric
# data['Battery %'] = data['Battery %'].str.replace('%', '').astype(float)

# # Label encode the "Employees" column
# le = LabelEncoder()
# data['Employees'] = le.fit_transform(data['Employees'])

# # Label the data (1 = notify, 0 = don't notify)
# data['notify'] = data['Battery %'].apply(lambda x: 1 if x <= 10 else 0)

# # Features and labels
# X = data[['Battery %', 'Employees']]  # Use encoded 'Employees' column
# y = data['notify']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train a decision tree classifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # For real-time data (Example inputs)
# current_battery_level = 9  # Current battery level
# current_employee = 'Employee 13'  # Assuming employee name

# # Encode the current employee's name using the same encoder
# current_employee_encoded = le.transform([current_employee])[0]

# # Prepare new data for prediction
# new_data = pd.DataFrame({
#     'Battery %': [current_battery_level],
#     'Employees': [current_employee_encoded]
# })

# # Predict notification
# should_notify = model.predict(new_data)

# if should_notify == 1:
#     print("Notify the user: Battery is below 10%!")
# else:
#     print("No need to notify the user.")





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('dummyDataset.csv')

# Remove '%' sign and convert 'Battery %' column to numeric
data['Battery %'] = data['Battery %'].str.replace('%', '').astype(float)

# Label encode the "Employees" column
le = LabelEncoder()
data['Employees_Encoded'] = le.fit_transform(data['Employees'])

# Label the data (1 = notify, 0 = don't notify)
data['notify'] = data['Battery %'].apply(lambda x: 1 if x <= 10 else 0)

# Features and labels
X = data[['Battery %', 'Employees_Encoded']]  # Use encoded 'Employees' column
y = data['notify']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Now, check the entire dataset to print employees with battery levels <= 10%
low_battery_employees = data[data['Battery %'] <= 10]

# Decode the employee names from the encoded labels
low_battery_employees['Employee_Name'] = le.inverse_transform(low_battery_employees['Employees_Encoded'])

# Print notifications for employees with battery levels <= 10%
for index, row in low_battery_employees.iterrows():
    print(f"Notify {row['Employee_Name']}: Battery is at {row['Battery %']}%, please charge your device!")



