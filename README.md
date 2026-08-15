# ✍️ Handwritten Digit Recognition

A browser-based handwritten digit recognition system powered by a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** and deployed entirely client-side using **TensorFlow.js**.

The application allows users to draw a digit from **0–9** and instantly see the model's prediction, confidence score, complete probability distribution across all ten classes, and recent prediction history.

## 🚀 Live Demo

**[Try the Live Demo](https://shabab991.github.io/handwritten-digit-recognition/)**

> No installation required — the model runs directly inside the browser.

---

## 📌 Features

* 🧠 **CNN-based digit classification**
* ✍️ Interactive HTML5 canvas for drawing digits
* ⚡ Real-time client-side inference with TensorFlow.js
* 📊 Complete **softmax probability distribution** for digits 0–9
* 🎯 Prediction confidence score
* 🕘 Prediction history for recent attempts
* 📱 Mouse and touch support
* ⚠️ Model loading error handling
* 🎨 Responsive dashboard-style interface
* 🔒 No drawing data is sent to a backend server

---

## 🧠 How It Works

The application follows a simple machine-learning inference pipeline:

```text
User draws digit
       ↓
HTML5 Canvas
       ↓
Resize to 28 × 28
       ↓
Normalize pixel values
       ↓
TensorFlow.js
       ↓
CNN Model
       ↓
Softmax Output
       ↓
Probability Distribution
       ↓
Predicted Digit + Confidence
```

The trained CNN expects an MNIST-style **28 × 28 grayscale image**.

The model produces ten softmax probabilities:

```text
P(0), P(1), P(2), ..., P(9)
```

The digit with the highest probability becomes the final prediction.

---

## 📊 Why Show the Full Probability Distribution?

Instead of displaying only the winning class, this project exposes the model's complete output distribution.

For example:

```text
Digit 0 → 0.1%
Digit 1 → 0.0%
Digit 2 → 0.2%
Digit 3 → 0.4%
Digit 4 → 0.1%
Digit 5 → 98.7%
Digit 6 → 0.1%
Digit 7 → 0.2%
Digit 8 → 0.1%
Digit 9 → 0.1%
```

This provides more insight into **model confidence and classification behavior** than a single predicted label.

---

## 🛠️ Tech Stack

| Technology         | Purpose                          |
| ------------------ | -------------------------------- |
| Python             | Model training                   |
| TensorFlow / Keras | CNN development                  |
| MNIST              | Training and evaluation dataset  |
| TensorFlow.js      | Browser-based inference          |
| JavaScript         | Prediction and application logic |
| HTML5 Canvas       | Handwritten digit input          |
| HTML5              | Application structure            |
| CSS3               | Responsive dashboard UI          |
| GitHub Pages       | Deployment                       |

---

## 🧪 Model

The neural network is a CNN designed for MNIST digit classification.

Architecture:

```text
Input: 28 × 28 × 1
        ↓
Convolution + Max Pooling
        ↓
Convolution + Max Pooling
        ↓
Dense Layer (128)
        ↓
Dense Layer (10)
        ↓
Softmax
```

The final layer contains **10 output classes**, representing digits 0 through 9.

---

## 📁 Project Structure

```text
handwritten-digit-recognition/
│
├── index.html
├── style.css
├── script.js
├── model.py
│
└── model/
    ├── model.json
    └── group1-4-shard1of1
│
└── README.md
```

---

## 🐛 Bugs Fixed

### 1. Touch input

The original implementation used:

```javascript
"ontouchstart"
```

with `addEventListener()`.

This was corrected to:

```javascript
"touchstart"
"touchmove"
"touchend"
```

This enables drawing on touch-enabled devices.

### 2. TensorFlow.js model loading

The deprecated:

```javascript
tf.loadModel()
```

was replaced with:

```javascript
tf.loadLayersModel()
```

### 3. Model loading error handling

The application now displays a visible status when the model successfully loads or fails to load.

### 4. Probability visualization

The application now displays the complete softmax output instead of only the highest-probability digit.

### 5. Prediction history

The latest six predictions are displayed so users can observe model behavior across multiple inputs.

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/Shabab991/handwritten-digit-recognition.git
```

Enter the project directory:

```bash
cd handwritten-digit-recognition
```

Start a local HTTP server:

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

A local server is recommended because browsers may restrict loading model files when the application is opened directly using a `file://` URL.

---

## 🌐 Deployment

The project is deployed using **GitHub Pages**.

Live application:

**https://shabab991.github.io/handwritten-digit-recognition/**

The model is loaded from the repository and inference happens directly in the user's browser.

---

## 🔐 Privacy

The application performs inference **client-side** using TensorFlow.js.

The handwritten canvas input does not need to be uploaded to a backend server for prediction.

---

## 🎯 Portfolio Value

This project demonstrates practical understanding of:

* Convolutional Neural Networks
* Image classification
* MNIST preprocessing
* Softmax probability distributions
* Model inference
* TensorFlow and Keras
* TensorFlow.js
* JavaScript
* Client-side machine learning
* Model evaluation concepts
* Interactive data visualization
* GitHub Pages deployment

It goes beyond simply displaying a predicted class by exposing the model's **probability distribution and prediction behavior**.

---

## 👨‍💻 Author

**Shabab Ahmad**

MCA Student | Aspiring Data Engineer | Data Analytics | ML/AI

GitHub: **https://github.com/Shabab991**

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub.
