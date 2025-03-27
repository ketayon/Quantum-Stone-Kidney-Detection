# 🧠 Quantum AI Kidney Stone Detection

This project is an end-to-end AI solution that integrates Quantum Machine Learning, Computer Vision, and Quantum Models to detect kidney stones from ultrasound images.
It leverages PegasosQSVC, and AerSimulator (local quantum simulator) for diagnosis.

---

## 🚀 Features

- ✅ Real Kidney Ultrasound Image Processing & Augmentation
- ✅ Quantum-Classical Hybrid Computation
- ✅ Local Simulation via Qiskit’s AerSimulator
- ✅ Live Web UI for Visualization & Prediction
- ✅ Upload your own ultrasound image and get quantum-based prediction (image size 256x256)
- ✅ CLI Support for Direct Model Execution
- ✅ Dockerized for Seamless Deployment 

---

## 🏗️ Solution Architecture

### 🔬 End-to-End Processing Pipeline

1. **Ultrasound Preprocessing**  
   - Converts uploaded images to grayscale 
   - Applies Gaussian blur and scaling
   - Extracts meaningful features

2. **Quantum Feature Extraction**  
   - Reduces data dimensionality using PCA  
   - Expands into 54-parameter quantum ansatz

3. **Quantum Model Training & Classification**  
   - Uses PegasosQSVC trained on classical features 
   - Performs classification using AerSimulator for expectation value

4. **Automated Workflow Execution**  
   - JobScheduler and WorkflowManager handle orchestration  
   - All jobs run locally, no cloud needed

5. **Visualization & Prediction Interfacen**  
   - Web UI shows uploaded scan, PCA plots, model score, and quantum prediction.  
   - Classifies uploaded image via quantum circuit  
   - Returns either:
      -- ✅ Kidney Stone Detected
      -- ✅ No Kidney Stone
      -- ❌ Invalid MRI image (if corrupt/empty)

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ketayon/Quantum-Kidney-Stone-Detection
cd Quantum-Kidney-Stone-Detection
```

### 2️⃣ **Setup Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate   # MacOS/Linux  
venv\Scripts\activate      # Windows  
```

### 3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## 🔥 Running the System  

### **1️⃣ CLI Mode**  
```bash
python interfaces/cli.py --model-score

--dataset-info     Show dataset statistics
--model-score      Show PegasosQSVC model accuracy
--predict-qsvc     Classify with PegasosQSVC model
```
✅ **Output Example:**  
`Quantum QSVC on the training dataset: 0.89`  
`Quantum QSVC on the test dataset: 0.82`  

---

### **2️⃣ Web Interface**  
```bash
python interfaces/web_app/app.py
```
```
🖥 Open in your browser:
http://127.0.0.1:5000/

✅ Web UI Features:

📤 Upload Kidney Ultrasound Images

🧠 Instant Classification via Quantum Circuit (simulated)

📊 PCA Plot + Confusion Matrix + Probability Histogram

🔬 Model Scores and Real-Time Visualization
```

---

## 🐳 Deploying with Docker  

### **1️⃣ Build Docker Image**  
```bash
docker build -t quantum-kidney-stone .
```

### **2️⃣ Run Container**  
```bash
docker run -p 5000:5000 quantum-kidney-stone
```

🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`**  

---

## 🛠️ Development & Testing  

### **Run PyTests**  
```bash
pytest tests/
```

---
```
## ☁️ No Cloud Dependency
🛑 Previously used IBM Quantum Cloud for real quantum backend.
✅ Now uses Qiskit AerSimulator to simulate all quantum circuits locally.
✅ Works offline, no internet or API key required.
```

---