Project Title

Fake News Detection Using Machine Learning and Deep Learning Techniques

Group Members

| Name          | Index Number  | Role                      |
|---------------|---------------|---------------------------|
| Group Leader  | CIT-24-01-0460| Logistic Regression & CNN |
| Member 02     | CIT-24-01-0227| Naive Bayes & LSTM        |
| Member 03     | CIT-24-01-0072| SVM & BERT                |

Problem Statement

The rapid spread of fake news through online news platforms and social media has become a major challenge in today's digital world. Misinformation can influence public opinion, create social unrest, and reduce trust in reliable information sources.

This project aims to develop an automated Fake News Detection system using Natural Language Processing (NLP) techniques and compare the performance of multiple Machine Learning and Deep Learning models in classifying news articles as Fake or Real.



# Dataset Information

##### Dataset Name: 
  *Fake and Real News Dataset*

##### Source: 
  * Kaggle

##### Classes:

  * Fake News
  * Real News

##### Features:

  * Title
  * News Text
  * Subject
  * Date
  * Label

##### Dataset Size:
  * Approximately 44,000 news articles.


Project Structure

```text
project-root/
│
├── data/
├── notebooks/
├── src/
├── models/
├── reports/
├── screenshots/
├── videos/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/Tiwanka0312/NLP_Group_39.git
cd NLP_Group_39
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

### Data Preprocessing

```bash
python src/preprocessing.py
```

### Train Models

```bash
python src/train.py
```

### Evaluate Models

```bash
python src/evaluate.py
```

### Run Application

```bash
python src/app.py
```

---

## Model Summary

### Machine Learning Models

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

### Deep Learning Models

* CNN (Convolutional Neural Network)
* LSTM (Long Short-Term Memory)
* BERT Transformer

### Feature Extraction

* TF-IDF Vectorization
* Text Preprocessing
* Tokenization
* Stop-word Removal
* Lemmatization

---

## Results Summary

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | TBD      |
| Naive Bayes         | TBD      |
| SVM                 | TBD      |
| CNN                 | TBD      |
| LSTM                | TBD      |
| BERT                | TBD      |

### Best Performing Model

To be determined after model evaluation.

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* TensorFlow / Keras
* Hugging Face Transformers
* Git & GitHub

---

## Repository

https://github.com/Tiwanka0312/NLP_Group_39
