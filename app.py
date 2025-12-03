import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Title
st.title("🎓 Student Performance Predictor")

# Load dataset
try:
    data = pd.read_csv("student_data.csv")
    st.write("### Dataset Preview")
    st.dataframe(data.head())
except FileNotFoundError:
    st.error("❌ Could not find 'student_data.csv'. Please upload the file.")
    st.stop()

# Ensure required columns exist
required_cols = ["Hours_Studied", "Attendance", "Assignments_Submitted", "Result"]
missing_cols = [col for col in required_cols if col not in data.columns]

if missing_cols:
    st.error(f"❌ Missing columns in dataset: {', '.join(missing_cols)}")
    st.stop()

# Features and Target
X = data[["Hours_Studied", "Attendance", "Assignments_Submitted"]]
y = data["Result"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test) * 100
st.subheader(f"📊 Model Accuracy: {accuracy:.2f}%")

# ===== Graph 1: Accuracy Donut Chart =====
st.write("### Model Accuracy (Donut Chart)")
fig1, ax1 = plt.subplots()
ax1.pie(
    [accuracy, 100 - accuracy],
    labels=["Accuracy", "Error"],
    colors=["green", "red"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.4)  # makes it a donut
)
ax1.axis("equal")
st.pyplot(fig1)

# ===== Graph 2: Pass vs Fail Distribution =====
st.write("### Pass vs Fail Distribution")
fig2, ax2 = plt.subplots()
data["Result"].value_counts().plot(kind="bar", color=["skyblue", "orange"], ax=ax2)
ax2.set_xlabel("Result")
ax2.set_ylabel("Number of Students")
st.pyplot(fig2)

# ===== User Input for Prediction =====
st.write("### 🔮 Predict Student Result")

hours = st.slider("Hours Studied", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
assignments = st.slider("Assignments Submitted", 0, 10, 5)

if st.button("Predict Result"):
    prediction = model.predict([[hours, attendance, assignments]])
    st.success(f"Predicted Result: {prediction[0]}")
