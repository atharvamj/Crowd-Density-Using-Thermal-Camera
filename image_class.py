import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

def preprocess_data(data):
    # Reshape and normalize the input data
    data = np.array([x.reshape(28, 28, 1) / 255.0 for x in data])
    return data

def create_cnn_model(input_shape, num_classes):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model

def run_cnn(train_image, train_label, test_image):
    # Preprocess the data
    train_data = preprocess_data(train_image)
    test_data = preprocess_data(test_image)

    # Convert labels to one-hot encoding
    label_binarizer = LabelBinarizer()
    train_label_one_hot = label_binarizer.fit_transform(train_label)

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        train_data, train_label_one_hot, test_size=0.2, random_state=42
    )

    # Define the CNN model
    input_shape = train_data[0].shape
    num_classes = len(label_binarizer.classes_)
    model = create_cnn_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data, label_binarizer.transform(test_label), verbose=2)

    # Print the test accuracy
    print(f'Test Accuracy: {test_accuracy}')

    # Save the predicted labels to a file
    # with open('project_xiaoq_cnn.txt', 'w') as file:
    #     file.write('\n'.join(map(str, predicted_labels)))
    #     file.flush()
    return True

if __name__ == "__main__":
    try:
        df = pd.read_pickle('C:/Users/mjdja/PycharmProjects/Crowd_Density/yolov4-deepsort/test100c5k_nolabel.pkl')  # training set path
        train_data = df['data'].values
        train_target = df['target'].values
        df = pd.read_pickle('C:/Users/mjdja/PycharmProjects/Crowd_Density/yolov4-deepsort/train100c5k.pkl')  # test set path
        test_data = df['data'].values
        test_label = df['target'].values
        info = run_cnn(train_data, train_target, test_data, test_label)
        if not info:
            print(sys.argv[0] + ": Return False")
    except RuntimeError:
        print(sys.argv[0] + ": A RuntimeError occurred")
    except Exception as e:
        print(sys.argv[0] + f": An exception occurred: {str(e)}")
