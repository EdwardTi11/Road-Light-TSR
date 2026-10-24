# 🚦 RoadLightTSR: Lightweight Traffic Sign Classification with TensorFlow

RoadLightTSR is a lightweight Convolutional Neural Network (CNN) for traffic sign classification built with TensorFlow/Keras and trained on the German Traffic Sign Recognition Benchmark (GTSRB).

Unlike larger transfer-learning approaches, RoadLightTSR prioritizes:

* Fast inference latency
* Low parameter count
* Minimal computational overhead
* Suitability for edge deployment

The project demonstrates that compact CNN architectures can achieve strong classification performance while maintaining real-time inference capabilities.

---

## 📁 Repository Structure

```text
RoadLightTSR/
│
├── ccir.ipynb
├── requirements.txt
├── README.md
├── LICENSE

```

---

## 📊 Results

| Metric                    | Value       |
| ------------------------- | ----------- |
| Test Accuracy             | **94.1%**  |
| Test Loss                 | **0.2987**  |
| Median Inference Latency  | **0.61 ms** |
| Total Parameters          | **44,715**  |
| Number of Classes         | **43**      |

---

## 🗂 Dataset

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** streamed through DeepLake.

### Dataset Details

* 43 traffic sign classes
* 39,209 training images
* 7,537 test images (subset of official one with 12,360 images)
* RGB images with varying resolutions
* Images resized to 26 × 26 before training

```python
train_ds = dl.load("hub://activeloop/gtsrb-train")
test_ds = dl.load("hub://activeloop/gtsrb-test")
```

---

## 🧠 Model Architecture

RoadLightTSR uses a compact CNN consisting of:

```text
Input (26×26×3)
│
├── Conv2D (32 filters, ReLU)
├── MaxPooling2D
│
├── Conv2D (32 filters, ReLU)
├── BatchNormalization
├── MaxPooling2D
│
├── Flatten
├── Dropout
│
└── Dense (43, Softmax)
```

### Parameter Breakdown

| Layer              | Parameters |
| ------------------ | ---------: |
| Conv2D             |        896 |
| Conv2D             |      9,248 |
| BatchNormalization |        128 |
| Dense              |     34,443 |
| **Total model params** | 44,715   |
| **Non-trainable** params: | 64    |
| **Trainable model params (model size)** | 44,651 |

---

## ⚙️ Training Configuration

| Parameter             | Value                           |
| --------------------- | ------------------------------- |
| Epochs (with Early Stopping)                | 23                              |
| Batch Size            | 64                             |
| Optimizer             | Adam                            |
| Scheduler             | Cosine Decay                    |
| Initial Learning Rate | 0.001                           |
| Loss Function         | Sparse Categorical Crossentropy |

---

## ⏱ Latency Benchmarking

Inference latency is measured after a warm-up phase using TensorFlow graph execution:

```python
compiled_model = tf.function(model)
```

---

## 📊 Confusion Matrix

The normalized error bar highlights class-wise performance and common misclassifications.


---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/RoadLightTSR.git
cd RoadLightTSR
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook ccir.ipynb
```

Then execute all notebook cells sequentially to reproduce training, evaluation, latency measurements, and visualizations.

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* DeepLake
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 🔮 Future Improvements

* TensorFlow Lite export
* INT8 quantization
* ONNX deployment
* Raspberry Pi benchmarking
* Real-time webcam traffic sign recognition

---

## 📄 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
