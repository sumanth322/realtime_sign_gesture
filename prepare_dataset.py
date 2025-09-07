import os
import numpy as np
from sklearn.model_selection import train_test_split

# Path where you saved your .npy gesture data
DATA_PATH = os.path.join('data', 'processed')

# Get all class labels automatically from folder names
actions = os.listdir(DATA_PATH)  # ['yes', 'no', 'hello', 'thanks']

X, y = [], []

# Loop through each action and load its .npy files
for idx, action in enumerate(actions):
    action_folder = os.path.join(DATA_PATH, action)
    for file in os.listdir(action_folder):
        if file.endswith('.npy'):
            data = np.load(os.path.join(action_folder, file))
            X.append(data)
            y.append(idx)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save dataset
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("✅ Dataset prepared and saved successfully!")
print("Classes:", actions)
