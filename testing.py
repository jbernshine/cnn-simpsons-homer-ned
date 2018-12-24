import tensorflow as tf
import os
import cv2

TESTDIR = "./testing"
CATEGORIES = ["homer", "ned"]


def prepare(filepath):
    IMG_SIZE = 70
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def test():
    model = tf.keras.models.load_model("simpsons-homer-ned.model")

    total_test = 0
    total_success = 0

    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(TESTDIR, category)
        num_test = 0
        num_success = 0

        for img in os.listdir(path):
            try:
                prediction = model.predict([prepare(os.path.join(path, img))])
                num_test += 1

                if int(prediction[0][0]) == class_num:
                    num_success += 1

            except Exception as e:
                pass

        percent_success = num_success / num_test
        print(category, ":", " Number Tested: ", num_test, " Successful predictions: ", num_success,
              " Percent Success: ", percent_success)
        total_test += num_test
        total_success += num_success

    percent_success = total_success / total_test
    print("Total Number Tested: ", total_test, " Total successful predictions: ", total_success,
          " Total percent Success: ", percent_success)

if __name__ == '__main__':
    test()

