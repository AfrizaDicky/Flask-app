import cv2
import numpy as np
from keras.models import load_model
from yoloface import face_analysis
import logging
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from flask import Flask, render_template, Response
import os

# Ganti dengan token API bot Anda
TOKEN = "7259308600:AAF6r_3rn2U1OVb07e6hy-yvM_TyJh6gLZI"
CHAT_ID = "6849355351"  # Ganti dengan chat_id yang Anda dapatkan

# Load models
try:
    model_age = load_model('Models/model_age.hdf5')
    model_gender = load_model('Models/model_gender.hdf5')
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise

# Labels
label_gender = ['Male', 'Female']

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Fungsi untuk menangani command /start
def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    update.message.reply_text(f'Hello, {user.first_name}! Send me a message and I will reply.')

# Fungsi untuk menangani pesan teks
def echo(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'You said: {update.message.text}')

# Fungsi untuk menangani error
def error(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Update {update} caused error {context.error}')

# Buat bot instance
bot = Bot(TOKEN)

# Buat updater dan pass token bot Anda
updater = Updater(TOKEN)

# Dapatkan dispatcher untuk mendaftarkan handler
dispatcher = updater.dispatcher

# Mendaftarkan command handler untuk /start
dispatcher.add_handler(CommandHandler("start", start))

# Mendaftarkan handler untuk menangani pesan teks
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

# Log semua error
dispatcher.add_error_handler(error)

# Start the Bot
updater.start_polling()

# detect video
def gen_frames():
    esp32_url = 'http://172.20.14.123/cam-lo.jpg'  # Ganti dengan URL streaming ESP32-CAM Anda
    cap = cv2.VideoCapture(esp32_url)
    if not cap.isOpened():
        logging.error("Error opening video capture")
        return

    face = face_analysis()

    while True:
        ret, img = cap.read()
        if not ret:
            logging.error("Failed to grab frame")
            break

        img = cv2.flip(img, 1)
        _, box, _ = face.face_detection(frame_arr=img, frame_status=True, model='tiny')

        for (x, y, w, h) in box:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_detect = cv2.resize(img[y:y + h, x:x + w], (50, 50)).reshape(1, 50, 50, 3)

            # Detect Age
            age = np.round(model_age.predict(img_detect / 255.0))[0][0]

            # Detect Gender
            gender_arg = np.round(model_gender.predict(img_detect / 255.0)).astype(np.uint8)
            gender = label_gender[gender_arg[0][0]]

            # Save the detected face image
            face_img = img[y:y + h, x:x + w]
            temp_img_path = 'temp.jpg'
            cv2.imwrite(temp_img_path, face_img)

            # Send message to Telegram
            bot.send_message(chat_id=CHAT_ID, text=f"Detected person: Age = {age}, Gender = {gender}")
            # bot.send_message(chat_id=CHAT_ID, text=f"Web Live Monitoring : /link")
            bot.send_photo(chat_id=CHAT_ID, photo=open(temp_img_path, 'rb'))

            # Remove the temporary image file
            os.remove(temp_img_path)

            # Draw
            cv2.putText(img, f'Age: {age}, {gender}', (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/halaman2')
def second():
    return render_template('camera.html')

@app.route('/halaman3')
def third():
    return render_template('design.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    logging.info("Starting video detection...")
    app.run(host='0.0.0.0', port=5000)
    updater.idle()
