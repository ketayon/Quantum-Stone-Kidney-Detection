# 🧠 Quantum AI Kidney Stone Detection

This project is an **end-to-end solution** that integrates **Quantum AI, Computer Vision, and Hybrid Quantum-Classical Models** to detect **kidney stones from ultrasound images**. It leverages **Quantum Kernel Learning, Quantum Support Vector Classifiers (QSVC), and Neural Networks** for enhanced medical diagnosis.

---

## 🚀 Features

- ✅ Real Kidney Ultrasound Image Processing & Augmentation  
- ✅ Quantum-Classical Hybrid Computation  
- ✅ Automated Workflow & IBM Quantum Cloud Integration  
- ✅ Live Web UI for Visualization & Prediction  
- ✅ Upload your own ultrasound image and get **quantum-based prediction**  
- ✅ CLI Support for Direct Model Execution  
- ✅ Dockerized for Seamless Deployment  

---

## 🏗️ Solution Architecture

### 🔬 End-to-End Processing Pipeline

1. **Ultrasound Preprocessing**  
   - Load ultrasound images and enhance them with intensity scaling.  
   - Convert scans into optimal feature vectors for Quantum AI.

2. **Quantum Feature Extraction**  
   - Reduces ultrasound data dimensionality using **PCA** or block averaging.  
   - Encodes optimized data into **Quantum Kernel Circuits**.

3. **Quantum Model Training & Classification**  
   - Uses **PegasosQSVC** trained with **Fidelity Quantum Kernel** on IBM Quantum.  
   - Hybrid **Quantum + Classical ML** improves medical diagnosis accuracy.

4. **Automated Workflow Execution**  
   - JobScheduler and WorkflowManager handle distributed computation.  
   - Executes on real IBM Quantum Hardware (or local simulator fallback).

5. **Real-Time Visualization & Quantum Prediction**  
   - UI shows **kidney scan, PCA visualization, model confidence, and confusion matrix**.  
   - 📤 **Upload custom ultrasound image → get prediction from a quantum circuit**  
   - ❗️If the uploaded image is invalid or too uniform, a warning will appear.

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
🖥 Open in your browser:
👉 http://127.0.0.1:5000/

UI Features:
🖼 View real ultrasound samples
📊 Visualize PCA, confusion matrix, and probability distributions
⚡ Get Quantum Model Performance Scores
📤 Upload your own ultrasound image for classification
Quantum circuit will process and return one of:
✅ "Kidney Stone Detected"
✅ "No Kidney Stone Detected"
❌ "Invalid image" if image is blank, noisy, or not kidney ultrasound
  

---

## 🐳 Deploying with Docker  

### **1️⃣ Build Docker Image**  
```bash
docker build -t quantum-kidney-stone .
```

### **2️⃣ Run Container**  
```bash
docker run -p 5000:5000 quantum-kidney-stone

if echo "QISKIT_IBM_TOKEN=your_ibm_quantum_token_here" > .env
docker run --env-file .env -p 5000:5000 quantum-kidney-stone
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

## 💼 IBM Quantum Cloud Integration  

**Setup IBM Quantum Account**  
1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)  
2. Get your API **Token** from **My Account**  
3. Set it in your environment:  
```bash
export QISKIT_IBM_TOKEN="your_ibm_quantum_token"
```

---