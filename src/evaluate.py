from sklearn.metrics import precision_score, recall_score


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))