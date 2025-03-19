# AI-Based Smart Parking System

## 📌 Project Overview
The **AI-Based Smart Parking System** is an intelligent parking management solution that utilizes **computer vision and AI** to detect vehicle license plates, record entry and exit times, and calculate parking fees automatically.

## 🚀 Features
- **License Plate Detection & Recognition** using OpenCV and Tesseract OCR.
- **Automated Parking Entry & Exit** logging with timestamps.
- **SQLite Database Integration** for storing records.
- **Web Dashboard (Flask-based)** to view and manage parking data.
- **Automated Fee Calculation** (₹10 per hour).

## 🛠️ Technologies Used
- **Python** (Core Programming Language)
- **OpenCV** (License Plate Detection)
- **Tesseract OCR** (Text Recognition)
- **SQLite** (Database Management)
- **Flask** (Web Framework)

## 📂 Folder Structure
```
📦 AI-Based-Smart-Parking-System
├── 📁 model
│   ├── haarcascade_russian_plate_number.xml
├── 📁 plates
│   ├── (Saved license plate images)
├── 📄 AI_parking1.py
├── 📄 requirements.txt
├── 📄 README.md
```

## 📦 Installation & Setup
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Purush-cnx/AI-Based-Smart-Parking-System.git
   cd AI-Based-Smart-Parking-System
   ```
2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Ensure Tesseract is Installed** and update its path in `AI_parking1.py` if necessary.
4. **Run the Application:**
   ```sh
   python AI_parking1.py
   ```
5. **Access the Web Dashboard:**
   - Open: `http://127.0.0.1:5000/view` (View parking records)
   - Open: `http://127.0.0.1:5000/clear` (Clear records)

## 🔥 Future Enhancements
- Upgrade **license plate detection model** from Haar cascade to **YOLOv8**.
- Improve OCR accuracy using **EasyOCR or PaddleOCR**.
- Implement **user authentication** for web access security.
- Migrate database from SQLite to **MySQL/PostgreSQL** for scalability.

## 🤝 Contribution
Feel free to fork this repository, create pull requests, and contribute!

## 📜 License
This project is licensed under the **MIT License**.

---
👨‍💻 Developed by **K Purushotham**
