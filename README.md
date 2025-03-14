# 🧠 Quantum AI Kidney Stone Detection

This project is an **end-to-end solution** that integrates **Quantum AI, Computer Vision, and Hybrid Quantum-Classical Models** to detect **kidney stones from ultrasound images**. It leverages **Quantum Kernel Learning, Quantum Support Vector Classifiers (QSVC), and Neural Networks** for enhanced medical diagnosis.

## 🚀 Features  

- **Real Kidney Ultrasound Image Processing & Augmentation**  
- **Quantum-Classical Hybrid Computation**  
- **Automated Workflow & IBM Quantum Cloud Integration**  
- **Live Web UI for Ultrasound Visualization & Prediction**  
- **CLI Support for Direct Model Execution**  
- **Dockerized for Seamless Deployment**  

---

## 🏗 **Solution Architecture**  

### 🔬 **End-to-End Processing Pipeline**  

1. **Ultrasound Preprocessing**  
   - Load ultrasound images and enhance them with intensity scaling.  
   - Convert scans into optimal feature vectors for Quantum AI.  

2. **Quantum Feature Extraction**  
   - Reduces ultrasound data dimensionality using **PCA**.  
   - Encodes optimized data into **Quantum Kernel Circuits**.  

3. **Quantum Model Training & Classification**  
   - Uses **Quantum Support Vector Classifiers (QSVC)** trained on IBM Quantum Cloud.  
   - Hybrid **Quantum + Classical ML** improves kidney stone detection accuracy.  

4. **Automated Workflow Execution**  
   - **JobScheduler & WorkflowManager** distribute quantum-classical computations.  
   - IBM **Quantum Backend** executes feature processing.  

5. **Real-Time Visualization & Prediction**  
   - Web UI provides live ultrasound scan visualization.  
   - Users can **upload images, analyze quantum predictions, and visualize kidney stone detection probabilities**.  

---

## 🏗 **Installation Guide**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-repo/Quantum-Kidney-Stone-Detection
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
```
✅ **Output Example:**  
`Quantum QSVC on the training dataset: 0.89`  
`Quantum QSVC on the test dataset: 0.82`  

---

### **2️⃣ Web Interface**  
```bash
python interfaces/web_app/app.py
```
🖥 **Access Web App:**  
🔍 Open **`http://127.0.0.1:5000/`** in a browser.  
✔ Upload and analyze ultrasound images  
✔ View Quantum Model Predictions  
✔ Visualize Ultrasound Scans and Kidney Stone Probability Heatmaps  

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