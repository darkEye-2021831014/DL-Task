# Heart Disease Classifier — Logistic Regression from Scratch

A manual implementation of binary logistic regression using stochastic gradient descent (SGD), built without any ML training libraries. Trained and evaluated on the UCI Heart Disease dataset.

---

## Dataset

**Source:** UCI Heart Disease Dataset (`HeartDisease.csv`)  
**Samples:** 1025 patients  
**Features:** 13 clinical attributes  
**Target:** `1` = heart disease present, `0` = absent

| Feature    | Description                                |
| ---------- | ------------------------------------------ |
| `age`      | Age in years                               |
| `sex`      | Sex (1 = male, 0 = female)                 |
| `cp`       | Chest pain type (0–3)                      |
| `trestbps` | Resting blood pressure (mm Hg)             |
| `chol`     | Serum cholesterol (mg/dl)                  |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg`  | Resting ECG results (0–2)                  |
| `thalach`  | Maximum heart rate achieved                |
| `exang`    | Exercise induced angina (1 = yes)          |
| `oldpeak`  | ST depression induced by exercise          |
| `slope`    | Slope of peak exercise ST segment          |
| `ca`       | Number of major vessels (0–3)              |
| `thal`     | Thalassemia type (0–2)                     |

---

## How It Works

No `sklearn` model is used for training. The classifier is implemented entirely with `numpy`:

1. **Sigmoid activation** — squashes output to (0, 1) probability
2. **Forward pass** — `z = dot(x, w) + b`, `a = sigmoid(z)`
3. **Loss** — Mean Squared Error: `(a − y)²`
4. **Backward pass** — manual gradient computation
5. **Weight update** — `w -= eta * 2 * error * x`, `b -= eta * 2 * error`

---

## Key Design Decisions

**Sigmoid Clipping** — `z` is clipped to `[-500, 500]` before `exp` to prevent numerical overflow, which has no effect on the output since `sigmoid(±500) ≈ 0 or 1` already.

---

## Hyperparameters

| Parameter             | Value  | Notes                                                              |
| --------------------- | ------ | ------------------------------------------------------------------ |
| `eta` (learning rate) | `0.01` | Higher values cause overshooting; lower values converge too slowly |
| `epochs`              | `500`  | Sufficient for convergence; diminishing returns beyond this        |
| `test_size`           | `0.2`  | 80/20 train/test split                                             |
| `random_state`        | `42`   | Fixed seed for reproducibility                                     |

---

## Results

| Metric    | Value                                 |
| --------- | ------------------------------------- |
| Accuracy  | ~60–75%                               |
| Precision | per class (see classification report) |
| Recall    | per class (see classification report) |
| F1-Score  | per class (see classification report) |

Full classification report, and per-epoch loss table are printed inside the notebook.
