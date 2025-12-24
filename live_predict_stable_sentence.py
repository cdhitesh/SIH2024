import cv2
import numpy as np
import tensorflow as tf
import time
import sqlite3
from datetime import datetime

# Load trained model
model = tf.keras.models.load_model("sign_language_interpreter_model.h5")

# Class labels (0-based index)
class_labels = [
    "1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z"
]

IMG_SIZE = 64
CONF_THRESHOLD = 0.6
SPACE_DELAY = 1.5
MIN_STABLE_FRAMES = 6

current_word = ""
sentence = ""

last_predicted = ""
stable_count = 0
accepted_letter = ""

last_confident_time = time.time()
space_added = False

def save_sentence_to_db(sentence):
    conn = sqlite3.connect("sign_language.db")
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        "INSERT INTO conversations (timestamp, sentence) VALUES (?, ?)",
        (timestamp, sentence)
    )

    conn.commit()
    conn.close()

cap = cv2.VideoCapture(0)

print("Webcam started")
print("Hold sign steady for letter")
print("Pause hand to insert space")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)

    current_time = time.time()

    if confidence > CONF_THRESHOLD:
        predicted_letter = class_labels[class_index]
        last_confident_time = current_time

        if predicted_letter == last_predicted:
            stable_count += 1
        else:
            stable_count = 1
            last_predicted = predicted_letter

        if stable_count == MIN_STABLE_FRAMES:
            if predicted_letter != accepted_letter:
                current_word += predicted_letter
                accepted_letter = predicted_letter
                space_added = False
                print("Sentence:", sentence + current_word)

    else:
        if current_word and not space_added and current_time - last_confident_time > SPACE_DELAY:
            sentence += current_word + " "
            current_word = ""
            accepted_letter = ""
            last_predicted = ""
            stable_count = 0
            space_added = True
            print("Sentence:", sentence)

    cv2.putText(frame, f"Word: {current_word}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

final_sentence = sentence + current_word
print("\nFinal sentence:", final_sentence)

if final_sentence.strip():
    save_sentence_to_db(final_sentence)
    print("Sentence saved to database.")
