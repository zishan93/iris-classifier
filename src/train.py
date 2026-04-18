import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os

# Create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# 1. Load the flower data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# 2. Train the AI model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. Save the model (Tutor requirement!)
joblib.dump(model, 'outputs/model.joblib')

# 4. Create and save the Confusion Matrix (Tutor requirement!)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Iris Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')

print("Model training complete. Artifacts saved to /outputs.")