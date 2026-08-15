# ✍️ Handwritten Digit Recognition

A browser-based **Handwritten Digit Recognition** application powered by a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** and deployed entirely client-side using **TensorFlow.js**.

Draw a digit from **0–9** and get an instant prediction with the model's **confidence score, complete softmax probability distribution, and prediction history** — all directly inside the browser.

## 🚀 Live Demo

### 👉 [Try Handwritten Digit Recognition](https://shabab991.github.io/handwritten-digit-recognition/)

> No installation required. The trained model runs directly in the browser using TensorFlow.js.

---

## ✨ Features

* 🧠 **CNN-based handwritten digit classification**
* ✍️ Interactive HTML5 Canvas for drawing digits
* ⚡ Real-time **client-side ML inference**
* 📊 Complete **softmax probability distribution** for all 10 classes
* 🎯 Prediction confidence score
* 🕘 Last 6 predictions stored in prediction history
* 📱 Mouse and touch-screen support
* ⚠️ Model loading error handling
* 🎨 Responsive dashboard-style UI
* 🔒 No backend required for inference
* ☁️ Deployed with GitHub Pages

---

## 🧠 How It Works

The application follows a complete image-classification inference pipeline:

```text
User Draws Digit
       │
       ▼
HTML5 Canvas
       │
       ▼
Resize to 28 × 28
       │
       ▼
Pixel Normalization
       │
       ▼
TensorFlow.js
       │
       ▼
CNN Model
       │
       ▼
Softmax Output
       │
       ▼
10 Class Probabilities
       │
       ▼
Prediction + Confidence
```

The CNN expects an MNIST-style **28 × 28 grayscale image**.

The final layer produces ten probabilities:

```text
P(0), P(1), P(2), P(3), P(4)
P(5), P(6), P(7), P(8), P(9)
```

The class with the highest probability is selected as the predicted digit.

---

## 📊 Probability Distribution

Instead of displaying only the predicted digit, the application visualizes the **complete model output distribution**.

Example:

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

This makes the application more informative than a basic "digit recognizer" because it exposes how confident the classifier is across **all possible classes**.

---

## 🧪 CNN Model Architecture

The project uses a convolutional neural network designed for MNIST digit classification.

```text
Input
28 × 28 × 1
     │
     ▼
Convolution Layer
     │
     ▼
Max Pooling
     │
     ▼
Convolution Layer
     │
     ▼
Max Pooling
     │
     ▼
Flatten / Dense
     │
     ▼
Dense Layer
128 Units
     │
     ▼
Output Layer
10 Units
     │
     ▼
Softmax
```

The output layer contains **10 classes**, representing digits **0 through 9**.

---

## 🛠️ Tech Stack

| Technology         | Purpose                       |
| ------------------ | ----------------------------- |
| Python             | Model training                |
| TensorFlow / Keras | CNN development               |
| MNIST              | Training dataset              |
| TensorFlow.js      | Browser-based model inference |
| JavaScript         | Application logic             |
| HTML5 Canvas       | Digit drawing                 |
| HTML5              | Web application structure     |
| CSS3               | Responsive UI                 |
| Git & GitHub       | Version control               |
| GitHub Pages       | Deployment                    |

---

## 📁 Project Structure

```text
handwritten-digit-recognition/
│
├── index.html
├── style.css
├── script.js
├── model.py
├── README.md
│
└── model/
    ├── model.json
    └── group1-4-shard1of1
```

---

## 🐛 Improvements & Bugs Fixed

### 📱 1. Mobile Touch Input

The original implementation registered the wrong touch event names:

```javascript
"ontouchstart"
```

when using `addEventListener()`.

This was corrected to:

```javascript
"touchstart"
"touchmove"
"touchend"
```

As a result, digit drawing now works with touch-enabled devices.

---

### ⚙️ 2. TensorFlow.js Model Loading

The deprecated model-loading API:

```javascript
tf.loadModel()
```

was replaced with:

```javascript
tf.loadLayersModel()
```

This provides compatibility with modern TensorFlow.js usage.

---

### ⚠️ 3. Model Loading Error Handling

The application now handles model-loading failures with:

* Visible status messages
* Console error logging
* Graceful prediction handling while the model is loading

---

### 📊 4. Full Probability Distribution

The original application displayed only the winning prediction.

The enhanced version exposes the complete **10-class softmax output**, making the model's confidence and classification behavior easier to understand.

---

### 🕘 5. Prediction History

The application stores the latest **six predictions** so multiple recognition attempts can be compared during a session.

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shabab991/handwritten-digit-recognition.git
```

### 2. Enter the project directory

```bash
cd handwritten-digit-recognition
```

### 3. Start a local HTTP server

```bash
python -m http.server 8000
```

### 4. Open the application

```text
http://localhost:8000
```

> A local HTTP server is recommended instead of opening `index.html` directly with `file://`, because browsers can restrict local loading of model files.

---

## 🌐 Deployment

The application is deployed using **GitHub Pages**.

### Live Application

**https://shabab991.github.io/handwritten-digit-recognition/**

The trained TensorFlow.js model is loaded from the repository and inference is performed directly in the user's browser.

No Python backend is required to run predictions.

---

## 🔐 Privacy

The application performs inference **client-side** using TensorFlow.js.

The handwritten drawing does not need to be uploaded to a backend server for prediction.

```text
Your Browser
     │
     ├── Canvas Input
     │
     ├── TensorFlow.js
     │
     └── CNN Model
             │
             ▼
        Prediction
```

---

## 🎯 Portfolio Highlights

This project demonstrates practical understanding of:

* Convolutional Neural Networks
* Image classification
* MNIST preprocessing
* TensorFlow / Keras
* TensorFlow.js
* Softmax probability distributions
* Model inference
* JavaScript
* HTML5 Canvas
* Responsive UI development
* Client-side machine learning
* GitHub Pages deployment
* Basic model evaluation concepts

Rather than displaying only a predicted label, the application exposes the model's **complete probability distribution**, providing a clearer view of classifier confidence and behavior.

---

## 🔮 Future Improvements

Potential future enhancements:

* 📈 Training accuracy and loss visualization
* 🔥 Confusion matrix
* 📊 Precision, recall and F1-score
* 🧪 Model performance dashboard
* 🌓 Light / dark theme switch
* 📥 Export prediction history
* 📱 Progressive Web App support
* 🧠 Model version comparison
* 📊 Interactive prediction analytics

---

## 👨‍💻 Author

**Shabab Ahmad**

MCA Student | Aspiring Data Engineer | Data Analytics | ML/AI

GitHub: **https://github.com/Shabab991**

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.
