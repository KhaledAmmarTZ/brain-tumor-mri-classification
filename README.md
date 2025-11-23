# 🧠 Brain Tumor MRI Classification  
### *Custom CNN, VGG16, ResNet50 & DenseNet121 (Fine-Tuned)*

This project implements **binary classification (Tumor vs. No Tumor)** using four deep-learning models:

- **Custom CNN (Baseline Model)**
- **VGG16 – Fine-Tuned**
- **ResNet50 – Fine-Tuned**
- **DenseNet121 – Fine-Tuned (Best Performing Model)**

The models are trained and evaluated on a curated brain MRI dataset containing **8,277 training images** and **1,816 testing images**, organized into *Tumor* and *No Tumor* classes.

This repository includes complete training notebooks, evaluation scripts, confusion matrix generation, saved model weights, and single-image inference support.

---

# 📘 Google Colab Training Notebook  
Paste your notebook link here:

🔗 **Colab Notebook:**  
https://colab.research.google.com/YOUR_LINK_HERE

(Required by instructor)

---

# 📂 Dataset Structure  

Dataset must be arranged as follows:

BrainTumor/
Training/
Tumor/
No_Tumor/
Testing/
Tumor/
No_Tumor/

yaml
Copy code

Dataset Source (Mendeley DOI): **10.17632/c9rt8d6zrf.1**

---

# 🏗️ Project Structure  

BrainTumor-MRI-Classification/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│ ├── Custom_CNN.ipynb
│ ├── VGG16_FineTuned.ipynb
│ ├── ResNet50_FineTuned.ipynb
│ ├── DenseNet121_FineTuned.ipynb
│
├── models/
│ ├── custom_cnn.h5
│ ├── vgg16_finetuned.h5
│ ├── resnet50_finetuned.h5
│ └── densenet121_finetuned.h5
│
├── utils/
│ ├── dataset_inspector.py
│ ├── inference_single_image.py
│ ├── preprocess.py
│ └── plot_training.py
│
├── sample_input/
│ └── brain_mri_sample.jpg
│
└── sample_output/
├── accuracy_curve.png
├── loss_curve.png
└── confusion_matrix.png

yaml
Copy code

---

# 🧪 Models Used

## 1️⃣ Custom CNN (Baseline)

```python
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
2️⃣ VGG16 (Fine-Tuned)
python
Copy code
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
3️⃣ ResNet50 (Fine-Tuned)
python
Copy code
base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
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
4️⃣ DenseNet121 (Fine-Tuned) — ⭐ Best Model
python
Copy code
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
📊 Evaluation Code (Accuracy, Loss & Confusion Matrix)
python
Copy code
loss, acc = model.evaluate(test_gen)
print("Test Accuracy:", acc)
print("Test Loss:", loss)

y_true = test_gen.classes
y_pred = (model.predict(test_gen) > 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))
🔍 Single-Image Prediction
python
Copy code
img = cv2.imread("sample.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]
print("Tumor" if pred > 0.5 else "No Tumor")
📝 Results Summary
Model	Accuracy	Comment
DenseNet121	⭐ Highest	Best overall performance
ResNet50	High	Strong generalization
VGG16	Medium	Useful baseline TL model
Custom CNN	Lower	Good baseline benchmark

DenseNet121 performed the best across all metrics.

🎥 Presentation Demo (Required)
Add your project presentation video link here:

🔗 https://your-video-link-here


