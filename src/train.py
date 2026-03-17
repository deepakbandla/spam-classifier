import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

from email_loader import load_dataset
from pipeline import create_pipeline


ham_emails, spam_emails = load_dataset()
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) +
             [1] * len(spam_emails))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = create_pipeline()
X_train_transformed = pipeline.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_transformed, y_train)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(pipeline, open("models/pipeline.pkl", "wb"))