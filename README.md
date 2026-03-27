# Hybrid Fake News Detection using Logistic Regression + GenAI
A machine learning and LLM hybrid system to detect misinformation using Logistic Regression and Gemini 2.5 Flash.

# 🧠 Hybrid Fake News Detection System

🚀 **Detect fake news with speed, intelligence, and cost-efficiency**
---

## ⚡ What This Project Does
This system classifies news as **Real or Fake** using a **hybrid AI approach**:
* ⚡ **Logistic Regression + TF-IDF** → fast, reliable predictions
* 🧠 **Gemini LLM fallback** → handles complex & ambiguous cases
* 💰 **Cost-optimized** → LLM is used only when necessary
---
## 🎯 Key Results
* ✅ **~98% accuracy** on benchmark dataset
* ✅ **Highly stable model** (low variance across folds)
* ✅ **LLM usage reduced by ~80–90%** using confidence gating
* ✅ Handles **real-world ambiguous & unusual news cases**
---
## 🔥 Why This Project Stands Out
Most projects:
```text
Input → Model → Output ❌
```
This system:
```text
Input → ML → Confidence Check → LLM → Final Decision ✅
```
👉 Combines **speed + reasoning + cost-efficiency**
---
📂 Dataset
Fake and Real News Dataset (Kaggle)
Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
~44,000 news articles (Real + Fake)
Used for training and evaluation
---
## 🧩 System Architecture
```text
User Input
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Confidence Check
   ↓
High Confidence → ML Output
Low Confidence → Gemini LLM → Final Output
```
---

## 🧪 Example Behavior

| Input Type       | Output                 |
| ---------------- | ---------------------- |
| Normal news      | ⚡ ML handles instantly |
| Fake/sensational | ⚡ ML detects patterns  |
| Weird/rare claim | 🧠 LLM analyzes        |

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* TF-IDF
* Logistic Regression
* Google Gemini API

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

Set API key:

```bash
export GEMINI_API_KEY=your_key_here
```

Run:

```bash
python app.py
```

---

## 💡 Key Insight

> This project is not just a model — it's a **decision system** that balances accuracy, cost, and intelligence.

---

## 📌 Future Improvements

* Streamlit UI
* LLM response caching
* Source credibility scoring
* Real-time news API integration

---

## 👨‍💻 Author

Built as a **practical hybrid AI system** focusing on real-world constraints like cost, scalability, and reliability.

