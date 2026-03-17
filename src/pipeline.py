from sklearn.pipeline import Pipeline
from transformers import EmailToWordCounterTransformer
from transformers import WordCounterToVectorTransformer


def create_pipeline():
    pipeline = Pipeline([
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer())
    ])

    return pipeline