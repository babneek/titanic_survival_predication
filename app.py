import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")
st.write("Upload Titanic dataset (CSV) and explore survival predictions with Logistic Regression.")

# --- Upload Data ---
uploaded_file = st.file_uploader("📂 Upload your Titanic dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # --- EDA ---
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Survival Count")
        fig, ax = plt.subplots()
        sns.countplot(x="Survived", data=df, palette="Set2", ax=ax)
        ax.set_xticklabels(["Not Survived", "Survived"])
        st.pyplot(fig)

    with col2:
        st.write("Survival by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1", ax=ax)
        ax.set_xticklabels(["Female", "Male"])
        ax.legend(["Not Survived", "Survived"])
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.write("Survival by Passenger Class")
        fig, ax = plt.subplots()
        sns.countplot(x="Pclass", hue="Survived", data=df, palette="pastel", ax=ax)
        ax.legend(["Not Survived", "Survived"])
        st.pyplot(fig)

    with col4:
        st.write("Age Distribution by Survival")
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x="Age", hue="Survived", fill=True, common_norm=False, palette="muted", ax=ax)
        st.pyplot(fig)

    # --- Data Preprocessing ---
    st.header("⚙️ Data Preprocessing & Model Training")

    # Drop rows with missing essential values
    df = df.dropna(subset=["Age", "Embarked"])  

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Features and target
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
    X = df[features]
    y = df["Survived"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Evaluation ---
    st.header("📈 Model Evaluation")

    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc:.2%}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Not Survived", "Survived"], 
                yticklabels=["Not Survived", "Survived"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.rename(index={"0": "Not Survived", "1": "Survived"}, inplace=True)
    st.dataframe(report_df.style.background_gradient(cmap="Blues").format("{:.2f}"))

    # --- Prediction ---
    st.header("🔮 Try Prediction with Custom Input")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.2)
    embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

    if st.button("Predict Survival"):
        # Encode categorical
        sex_val = 0 if sex == "male" else 1
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0

        # Arrange input like training features
        input_data = pd.DataFrame([[
            pclass, sex_val, age, sibsp, parch, fare, embarked_q, embarked_s
        ]], columns=features)

        # Prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("🎉 The passenger would have SURVIVED!")
        else:
            st.error("☠️ The passenger would NOT have survived.")

else:
    st.info("👆 Please upload a Titanic CSV dataset to continue.")
