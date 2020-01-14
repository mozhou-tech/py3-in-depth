import tensorflow as tf
import autokeras as ak
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# Initialize the image classifier.
clf = ak.ImageClassifier(max_trials=10) # It tries 10 different models.
# Feed the image classifier with training data.
clf.fit(x_train, y_train)
# Predict with the best model.
predicted_y = clf.predict(x_test)
# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
