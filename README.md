# Diabetes Indicators Explorer

Predicts whether a person reports prediabetes or diabetes (`Diabetes_binary`) from 21 CDC health-survey indicators. The project has three parts: a Jupyter notebook for model development, a **FastAPI** backend that serves predictions, and a **Streamlit** app for single-record and CSV batch predictions.

> **Educational use only.** This is not a medical device. Predictions are not diagnoses and must not be used to make healthcare decisions.

## Results at a glance

Final model: **Gradient Boosting** (250 trees, learning rate 0.1, depth 2) with a tuned decision threshold of **0.23**. All numbers below are on a held-out test set of 50,736 people (7,069 with the condition, 43,667 without) that was not used for training or threshold selection.

| Metric | Value |
|---|---|
| ROC-AUC | **0.826** |
| Recall (cases found) | **60.4%** |
| Precision (flagged people who have the condition) | **38.0%** |
| F1 (class 1) | **0.467** |
| Specificity (healthy people correctly cleared) | 84.1% |
| Accuracy | 80.8% |
| Share of people flagged | 22.1% |

Accuracy is not the headline metric here: always predicting "no diabetes" is already 86.1% accurate because only 13.9% of people are positive. Judge the model by recall, precision and AUC.

## Model performance

### Confusion matrix (threshold 0.23)

| | Predicted no diabetes | Predicted diabetes |
|---|---|---|
| **Actually no diabetes** | 36,715 | 6,952 |
| **Actually diabetes** | 2,800 | 4,269 |

How to read it:
- The model finds about 6 in 10 real cases (4,269 of 7,069).
- Among the people it flags, 38% are real cases, about 2.7 times the 13.9% base rate.
- Among the people it does not flag, 92.9% are truly negative.
- It still misses about 4 in 10 cases, so a negative result is not reassurance.

### Why the threshold is 0.23 and not 0.5

The default threshold of 0.5 almost never flags anyone because the positive class is rare. The threshold was chosen to maximize F1 on out-of-fold predictions from the training set only.

| Threshold | Recall | Precision | F1 | Accuracy | People flagged |
|---|---|---|---|---|---|
| 0.50 (default) | 17.2% | 54.1% | 0.261 | 86.4% | 4.4% |
| **0.23 (tuned)** | **60.4%** | **38.0%** | **0.467** | 80.8% | 22.1% |

A lower threshold finds more cases at the cost of more false alarms. For another use case (for example, a cheap first-pass screen), a lower threshold with higher recall may be more appropriate.

### Model comparison

| Model | CV ROC-AUC (training set, 5-fold) | Test ROC-AUC | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8251 | 0.8211 | 0.372 | 0.605 | 0.460 |
| Random Forest | 0.8190 | 0.8150 | 0.352 | 0.635 | 0.453 |
| **Gradient Boosting** | **0.8305** | **0.8262** | 0.380 | 0.604 | **0.467** |

Precision, recall and F1 use each model's own F1-tuned threshold. Gradient Boosting was chosen for the highest cross-validated and test AUC. The differences are small, and Logistic Regression is nearly as good (0.005 lower AUC) and easier to interpret.


### Evaluation protocol

1. Stratified 80/20 train/test split (`random_state=42`).
2. Preprocessing inside the pipeline: `StandardScaler` on numeric and ordinal features, binary features passed through unchanged, plus the engineered feature `Age_BMI`.
3. Three models compared with 5-fold cross-validation on the training set.
4. For each model, the F1-optimal threshold is chosen on out-of-fold predictions (grid of 201 thresholds from 0 to 1) so the test set plays no part in the choice.
5. The test set is used only to report final results.

## Dataset

**Source:** [CDC Diabetes Health Indicators, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
**DOI:** [10.24432/C53919](https://doi.org/10.24432/C53919)

Based on the CDC Behavioral Risk Factor Surveillance System (BRFSS) telephone survey (2015 survey year). It has 253,680 records, 21 input features and no missing values.

Target (`Diabetes_binary`): `0` = no diabetes, `1` = prediabetes or diabetes (13.9% of records).

## How it works

```text
User
 |
 v
Streamlit interface
 |
 | 21 input features
 v
FastAPI
 |
 | creates Age_BMI = Age * BMI
 v
Saved preprocessing + Gradient Boosting pipeline
 |
 v
Prediction, class-1 probability, and threshold
 |
 v
Streamlit displays the result
```

Users never enter `Age_BMI`: the API calculates it from `Age` and `BMI` before calling the model. The saved artifact (`models/diabetes_model.joblib`) contains:

- `model`: the fitted pipeline, including preprocessing.
- `threshold`: the classification threshold (`0.23`).

A record is classified as class `1` when its predicted class-1 probability is greater than or equal to the threshold.

## Input features

The app and API expect these **21 columns** with the exact names and numeric encodings below (valid ranges are taken from the dataset).

| Feature | Type | Valid values |
|---|---|---|
| `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | Binary | `0` or `1` |
| `BMI` | Numeric | 12 to 98 |
| `MentHlth`, `PhysHlth` | Numeric (days in the past 30) | 0 to 30 |
| `GenHlth` | Ordinal | 1 (excellent) to 5 (poor) |
| `Age` | Ordinal, 5-year bands | 1 (18-24) to 13 (80 or older) |
| `Education` | Ordinal | 1 (never attended school) to 6 (college graduate) |
| `Income` | Ordinal | 1 (under $10k) to 8 ($75k or more) |

For CSV batch prediction:
- Include all 21 columns with the exact names and numeric values.
- Do not include `Diabetes_binary` or `Age_BMI`.
- The uploader checks for missing columns and missing values, but not for out-of-range values, so stay within the ranges above.

## Installation

Use the Python version the dependencies in `requirements.txt` were frozen with.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.txt
```

## Run locally

Start both services from the **project root**, in two terminals. Keep the API running while you use Streamlit.

**Terminal 1: FastAPI**

```bash
python -m uvicorn api.main:app --reload
```

- Health check: http://127.0.0.1:8000/health
- Interactive docs: http://127.0.0.1:8000/docs

**Terminal 2: Streamlit**

```bash
python -m streamlit run app/app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The app sends requests to `http://127.0.0.1:8000/predict`.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Confirms the API is running and reports the loaded threshold. |
| `POST` | `/predict` | Accepts one record with the 21 input features and returns a prediction. |

Example request (check the exact schema at `/docs`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"HighBP":1,"HighChol":1,"CholCheck":1,"BMI":31,"Smoker":1,"Stroke":0,
       "HeartDiseaseorAttack":0,"PhysActivity":1,"Fruits":1,"Veggies":1,
       "HvyAlcoholConsump":0,"AnyHealthcare":1,"NoDocbcCost":0,"GenHlth":3,
       "MentHlth":2,"PhysHlth":5,"DiffWalk":0,"Sex":1,"Age":9,"Education":4,"Income":5}'
```

The response contains:
- `prediction`: predicted class (`0` or `1`).
- `probability`: predicted probability of class `1`.
- `threshold`: threshold used for the classification.

## Project structure

```text
MFTProject/
├── notebook/                 # Model development and evaluation
├── data/
│   └── Diabetes_Health_Indicators.csv
├── models/
│   └── diabetes_model.joblib
├── api/
│   └── main.py
├── app/
│   └── app.py
├── screenshots/              # Application and evaluation screenshots
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducing the model

The notebook covers exploration, preprocessing, model comparison, threshold selection and evaluation.

1. Download the dataset from UCI and save it as `data/Diabetes_Health_Indicators.csv`.
2. Install the dependencies from `requirements.txt`.
3. Open the notebook from inside the `notebook/` folder (it reads `../data/...` and writes `../models/...`).
4. Run all cells. The last cells save the fitted pipeline and threshold to `models/diabetes_model.joblib`.

The API loads this saved artifact; it does not train the model at startup. All random seeds are fixed (`random_state=42`).

## Limitations and responsible use

- **Not a clinical tool.** It was not clinically validated, and its output is a model score, not a verified estimate of one person's risk.
- **Self-reported survey data.** The labels and features come from a telephone survey, and the target merges prediabetes and diabetes.
- **Associations are not causes.** For example, people who had a cholesterol check have a much higher diabetes rate (14.4% vs 2.5%), but that mostly reflects who gets tested and diagnosed, not a risk effect.
- **Population mismatch.** The dataset has 13.9% positives. In another population the prevalence, and therefore the meaning of a given probability, will differ.
- **Uneven performance.** Recall is low for young adults and about 40% of real cases are missed overall.
- **Possible optimism.** About 9.5% of the rows are exact duplicates, so test scores may be slightly optimistic.
- **Threshold choice.** The threshold of 0.23 maximizes F1 for this project. It is not a clinical decision threshold.
