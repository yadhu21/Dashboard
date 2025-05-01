

# 🏆 Sports Data Dashboard

A dynamic and interactive web application built with **Streamlit** that enables users to explore, visualize, and analyze football player performance using various statistical and machine learning techniques.

---

## 📊 Features

- **Interactive Data Filtering** by league
- **Performance Categorization** (Low, Average, High Scorer)
- **Visualizations**:
  - Metric-wise player performance (Bar Charts)
  - Correlation Matrix
  - Scatterplots (Goals vs xG)
  - League Distribution by Country
- **Descriptive Statistics** summary
- **Linear Regression Model**:
  - Predict goals based on xG, Shots, and Minutes
  - Live goal prediction using slider inputs
- **Decision Tree Classifier**:
  - Classify players into performance categories
  - Visual tree plot
- **Improvement Suggestions** for players based on stats
- **Download Filtered Data** as CSV

---

## ⚙️ Technologies Used

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 📂 File Structure

```plaintext
├── Data.csv
├── app.py               # Main Streamlit application
├── README.md            # Project documentation (this file)
```

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

---

## 📥 Sample Inputs for Prediction

You can enter values such as:
- xG: 15
- Shots: 120
- Minutes Played: 2000
- On Target: 60
- Matches Played: 25

The app will classify performance and suggest personalized improvements.

---

## 👨‍💻 Authors

- **Yadhu Krishnan C K** and team  
  Final Year Data Science Project

---

## 📌 License

This project is for educational purposes only. Contact the authors for reuse.
