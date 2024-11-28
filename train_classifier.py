import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

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
