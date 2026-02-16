# AI-Shampoo-Defect-Detector

This **Automated Quality Inspection System** leverages Deep Learning and Computer Vision to identify manufacturing flaws in shampoo bottles. Developed as a major project for **B.Tech CSE (AIML)**, the system utilizes a custom-trained **YOLOv8** model to detect dents and structural anomalies with high precision.

##  Key Features
* **Real-Time Detection**: Rapid inference using YOLOv8 for production-line speed.
* **Modern Web Dashboard**: Responsive Flask interface for easy image uploads and result visualization.
* **Automated Reporting**: Detailed inspection summary including defect classification and confidence scores.
* **Intelligent File Management**: Dynamic timestamping to ensure accurate session results and prevent browser caching issues.

##  Technical Stack
* **Language**: Python 3.8.10
* **Framework**: Flask
* **AI Engine**: YOLOv8 (Ultralytics)
* **Image Processing**: OpenCV, Pillow (PIL)
* **Frontend**: HTML5, CSS3 (Modern UI with slide-in animations)

##  Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/Akanshu05-AI/AI-Shampoo-Defect-Detector.git](https://github.com/Akanshu05-AI/AI-Shampoo-Defect-Detector.git)
   cd AI-Shampoo-Defect-Detector'''

##  Install Dependencies
* pip install -r requirements.txt

##  Run the Application
* python app.py
* **Access the system at http://127.0.0.1:5000**

## System Workflow
The system operates through a streamlined pipeline:
* Ingestion: User uploads an image through the Flask frontend.
* Analysis: The backend passes the image to the YOLOv8 model (best.pt).
* Annotation: The model generates bounding boxes and labels for detected defects.
* Display: The UI renders the original and analyzed images side-by-side with a summary report.

**Developed by: Akanshu Goel**
**Course: 3rd Year B.Tech CSE (AIML)**
**University: IILM University, Greater Noida**
