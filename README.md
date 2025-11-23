# 🧠 Brain Tumor MRI Classification  
### *Custom CNN, VGG16, MobileNetV2 & DenseNet121 (Fine-Tuned)*

This project implements **binary classification (Tumor vs. No Tumor)** using four deep-learning models:

---

## 🔹 Models Used

### 1️⃣ Custom CNN (Baseline Model)
- A simple convolutional neural network with three Conv-Pool blocks and fully connected layers.
- Provides a baseline for comparison.

### 2️⃣ VGG16 – Fine-Tuned
- Pre-trained on ImageNet, with last layers fine-tuned for brain MRI classification.

### 3️⃣ MobileNetV2 – Fine-Tuned
- Lightweight transfer learning model for faster training with good accuracy.

### 4️⃣ DenseNet121 – Fine-Tuned (**Best Performing Model**)
- Pre-trained DenseNet121, fine-tuned on the dataset.
- Achieves the highest accuracy and best generalization.

The models are trained and evaluated on a curated brain MRI dataset containing **8,277 training images** and **1,816 testing images**, organized into *Tumor* and *No Tumor* classes.

This repository includes complete training notebooks, evaluation scripts, confusion matrix generation, saved model weights, and single-image inference support.

---

## 📘 Google Colab Training Notebook  

**Colab Notebook:**  
[Colab Notebook Link](https://colab.research.google.com/drive/1T_7naloU-uTCWEOOtiS73PEbntXpSG0j?usp=sharing)

---

## 📂 Dataset Structure  

The dataset must be arranged as follows:
```
BrainTumor/
    Training/
        Tumor/
        No_Tumor/
    Testing/
        Tumor/
        No_Tumor/
```

**Dataset Source (Mendeley DOI):** [10.17632/c9rt8d6zrf.1](https://data.mendeley.com/datasets/c9rt8d6zrf/1)

---

## 🏗️ Project Structure  

```
BrainTumor-MRI-Classification/
│
├── README.md
├── requirements.txt
├── notebooks/
│   ├── Custom_CNN.ipynb
│   ├── VGG16_FineTuned.ipynb
│   ├── MobileNetV2_FineTuned.ipynb
│   └── DenseNet121_FineTuned.ipynb
├── models/
│   ├── custom_cnn.h5
│   ├── vgg16_finetuned.h5
│   ├── mobilenetv2_finetuned.h5
│   └── densenet121_finetuned.h5
├── utils/
│   ├── dataset_inspector.py
│   ├── inference_single_image.py
│   ├── preprocess.py
│   └── plot_training.py
├── sample_input/
│   └── brain_mri_sample.jpg
└── sample_output/
    ├── accuracy_curve.png
    ├── loss_curve.png
    └── confusion_matrix.png
```

---

## 🧪 Model Architectures

### 1️⃣ Custom CNN (Baseline)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
```

### 2️⃣ VGG16 (Fine-Tuned)
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

base = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = Flatten()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Unfreeze last 5 layers
for layer in base.layers[-5:]:
    layer.trainable = True
    
model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
```

### 3️⃣ MobileNetV2 (Fine-Tuned)
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune last 30 layers
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
```

### 4️⃣ DenseNet121 (Fine-Tuned) — ⭐ Best Model
```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

base = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Fine-tune last 40 layers
for layer in base.layers[-40:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
```

---

## 📊 Evaluation Code (Accuracy, Loss & Confusion Matrix)
```python
loss, acc = model.evaluate(test_gen)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

y_true = test_gen.classes
y_pred = (model.predict(test_gen) > 0.5).astype(int)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))
```

## 🔍 Single-Image Prediction
```python
import cv2
import numpy as np

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]
print("Tumor" if pred > 0.5 else "No Tumor")
```

---

## 📝 Results Summary
| Model       | Accuracy  | Comment                         |
| ----------- | --------- | ------------------------------- |
| DenseNet121 | ⭐ Highest | Best overall performance        |
| MobileNetV2 | High      | Fast and accurate               |
| VGG16       | Medium    | Good baseline transfer learning |
| Custom CNN  | Lower     | Benchmark model                 |

**DenseNet121 performed the best across all metrics.**

---

## 🎥 Presentation Demo (Required)

🔗 [Demo Video Placeholder — Add Link Here](https://your-video-link-here)
