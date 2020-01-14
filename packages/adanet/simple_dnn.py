import adanet
import tensorflow as tf

# Define the model head for computing loss and evaluation metrics.
from tensorflow_estimator.python.estimator.head.multi_class_head import MultiClassHead

def train_input_fn():
    pass

def eval_input_fn():
    pass

def predict_input_fn():
    pass

head = MultiClassHead(n_classes=10)

# Feature columns define how to process examples.
feature_columns = ...

# Learn to ensemble linear and neural network models.
estimator = adanet.AutoEnsembleEstimator(
    head=head,
    candidate_pool=lambda config: {
        "linear":
            tf.estimator.LinearEstimator(
                head=head,
                feature_columns=feature_columns,
                config=config,
                optimizer=...),
        "dnn":
            tf.estimator.DNNEstimator(
                head=head,
                feature_columns=feature_columns,
                config=config,
                optimizer=...,
                hidden_units=[1000, 500, 100])},
    max_iteration_steps=50)

estimator.train(input_fn=train_input_fn, steps=100)
metrics = estimator.evaluate(input_fn=eval_input_fn)
predictions = estimator.predict(input_fn=predict_input_fn)