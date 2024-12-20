import cv2
import pytesseract
import re
import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

# Path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\CNX\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Initialize Flask app
app = Flask(__name__)

# Database setup
def initialize_database():
    conn = sqlite3.connect('parking_system.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS parking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_plate TEXT NOT NULL,
                    entry_time TEXT,
                    exit_time TEXT,
                    duration REAL DEFAULT 0.0,
                    amount REAL DEFAULT 0.0
                 )''')
    conn.commit()
    conn.close()

# Update database schema to add missing columns
def update_database_schema():
    conn = sqlite3.connect('parking_system.db')
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE parking ADD COLUMN duration REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        print("Column 'duration' already exists.")
    try:
        c.execute("ALTER TABLE parking ADD COLUMN amount REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        print("Column 'amount' already exists.")
    conn.commit()
    conn.close()

# Path to Haar Cascade for license plate detection
harcascade = "model/haarcascade_russian_plate_number.xml"

# Camera configuration
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
count = 0
output_dir = "plates/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Preprocess image for better OCR results
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# Extract Indian license plate format
def extract_indian_plate(text):
    pattern = r'[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}'
    match = re.findall(pattern, text.upper())
    if match:
        return match[0]
    return None

# Assign entry to database
def assign_entry(license_plate):
    conn = sqlite3.connect('parking_system.db')
    c = conn.cursor()
    entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO parking (license_plate, entry_time) VALUES (?, ?)', (license_plate, entry_time))
    conn.commit()
    conn.close()
    print(f"Entry recorded for {license_plate} at {entry_time}")

# Process exit and calculate parking fee
def process_exit(license_plate):
    conn = sqlite3.connect('parking_system.db')
    c = conn.cursor()
    c.execute('SELECT id, entry_time FROM parking WHERE license_plate = ? AND exit_time IS NULL', (license_plate,))
    record = c.fetchone()
    if record:
        id, entry_time = record
        exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = (datetime.strptime(exit_time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
        amount = round(duration * 10, 2)  # ₹10 per hour
        c.execute('UPDATE parking SET exit_time = ?, duration = ?, amount = ? WHERE id = ?',
                  (exit_time, duration, amount, id))
        conn.commit()
        conn.close()
        print(f"Exit recorded for {license_plate}. Duration: {duration:.2f} hours, Amount: Rs {amount}")
    else:
        print("No entry found for this license plate or already exited.")

# Flask route to display the database
@app.route("/view")
def view_database():
    conn = sqlite3.connect('parking_system.db')
    c = conn.cursor()
    c.execute('SELECT license_plate, entry_time, exit_time, duration, amount FROM parking')
    records = c.fetchall()
    conn.close()
    return render_template('view.html', records=records)

# Flask route to clear the database
@app.route("/clear", methods=['POST'])
def clear_database():
    conn = sqlite3.connect('parking_system.db')
    c = conn.cursor()
    c.execute('DELETE FROM parking')
    conn.commit()
    conn.close()
    return redirect(url_for('view_database'))

# Main program loop
if __name__ == "__main__":
    initialize_database()
    update_database_schema()

    from threading import Thread

    def flask_thread():
        app.run(debug=True, use_reloader=False)

    Thread(target=flask_thread).start()

    while True:
        success, img = cap.read()

        plate_cascade = cv2.CascadeClassifier(harcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img_roi = preprocess_image(img[y: y + h, x: x + w])
                cv2.imshow("ROI", img_roi)

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            file_path = os.path.join(output_dir, f"scanned_img_{count}.jpg")
            cv2.imwrite(file_path, img_roi)

            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(img_roi, config=custom_config)
            plate_text = extract_indian_plate(text)

            if plate_text:
                print("Extracted License Plate (Indian Format):", plate_text)
                assign_entry(plate_text)

                # Save the detected text along with the image
                with open(os.path.join(output_dir, f"scanned_img_{count}_text.txt"), "w") as f:
                    f.write(plate_text)

            else:
                print("No valid Indian license plate detected.")

            count += 1

        elif cv2.waitKey(1) & 0xFF == ord('e'):
            print("Processing exit...")
            file_path = os.path.join(output_dir, f"scanned_img_exit_{count}.jpg")
            cv2.imwrite(file_path, img_roi)

            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(img_roi, config=custom_config)
            plate_text = extract_indian_plate(text)

            if plate_text:
                print("Detected License Plate for Exit (Indian Format):", plate_text)
                process_exit(plate_text)

            else:
                print("No valid Indian license plate detected.")

    cap.release()
    cv2.destroyAllWindows()
