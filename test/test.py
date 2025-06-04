from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
import os
sys.path.append(os.path.abspath('../deep-forest'))

from deepforest import CascadeForestClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

n_objects = X.shape[0]
n_features = X.shape[1]
print(n_objects, 'objects,', n_features, 'features')

print("X: ", X)
print("y: ", y)
print("labels: ", set([int(i) for i in y]))
print("Train set size: ", X_train.shape[0])
print("Test set size: ", X_test.shape[0])

model = CascadeForestClassifier(
    n_estimators=4,
    n_trees=100,
    predictor_kwargs={
        "name": "prf",
        "max_features": 0.5,
        "keep_proba": 0.1,
    },
    backend="custom",
    random_state=42,
    verbose=1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("\nTesting Accuracy: {:.3f} %".format(acc))