import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Get the raw data
raw_data = data_dict['data']
labels = data_dict['labels']

# Check data consistency
expected_length = 42  # 21 landmarks * 2 (x,y coordinates)
valid_data = []
valid_labels = []

print(f"Total samples before filtering: {len(raw_data)}")

# Filter out inconsistent data points
for i, data_point in enumerate(raw_data):
    if len(data_point) == expected_length:
        valid_data.append(data_point)
        valid_labels.append(labels[i])

print(f"Total samples after filtering: {len(valid_data)}")

# Convert to numpy arrays
data = np.array(valid_data)
labels = np.array(valid_labels)

# Create labels dictionary
unique_labels = np.unique(labels)
labels_dict = {i: label for i, label in enumerate(unique_labels)}
# Create reverse mapping for training
label_to_index = {label: i for i, label in labels_dict.items()}
# Convert string labels to numeric indices
numeric_labels = np.array([label_to_index[label] for label in labels])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, numeric_labels, test_size=0.2, shuffle=True, stratify=numeric_labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

# Save both model and labels dictionary
f = open('model.p', 'wb')
pickle.dump({'model': model, 'labels_dict': labels_dict}, f)
f.close()
