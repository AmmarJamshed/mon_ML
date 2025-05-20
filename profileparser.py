import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense
import openai

# Streamlit Page Config
st.set_page_config(page_title="Coursemon AutoML Lab", layout="wide", page_icon="🧠")

# Styling & Branding
st.markdown(
    """
    <style>
        body {
            background-color: #f4faff;
        }
        .stApp {
            background-image: url("https://coursemon.net/assets/bg-transparent.png");
            background-size: cover;
        }
        header, footer {visibility: hidden;}
        .coursemon-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #1f2e55;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Coursemon AutoML Lab")
st.caption("Make classification, regression, clustering or deep learning models without writing code!")

# GPT Business Tab
tab1, tab2 = st.tabs(["📊 Model Builder", "🧠 Business Objective"])

with tab2:
    st.subheader("Describe Your Business Objective")
    business_query = st.text_area("What do you want to predict or understand?")
    if st.button("Let GPT Help"):
        if business_query:
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                with st.spinner("Thinking..."):
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You're a helpful machine learning assistant."},
                            {"role": "user", "content": f"I want to: {business_query}. What should be the target variable? Should I use classification, regression, or clustering? Suggest preprocessing too."}
                        ]
                    )
                    st.success("Here's what GPT suggests:")
                    st.info(response.choices[0].message["content"])
            except Exception as e:
                st.error(f"Error using GPT: {e}")

# Model Builder Tab
with tab1:
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview", df.head())

        # Fill missing values (basic logic)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        target_column = st.selectbox("🎯 Choose Target Variable (for supervised tasks)", ["None"] + list(df.columns))

        ml_type = st.selectbox("📌 Select Task Type", [
            "Supervised - Classification",
            "Supervised - Regression",
            "Unsupervised - Clustering",
            "Deep Learning (Binary)"
        ])

        model = None

        if ml_type == "Supervised - Classification":
            model_choice = st.selectbox("Choose a Model", ["Random Forest Classifier", "Logistic Regression"])
            model = RandomForestClassifier() if model_choice == "Random Forest Classifier" else LogisticRegression()

        elif ml_type == "Supervised - Regression":
            model_choice = st.selectbox("Choose a Model", ["Random Forest Regressor"])
            model = RandomForestRegressor()

        elif ml_type == "Unsupervised - Clustering":
            clusters = st.slider("Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=clusters)

        elif ml_type == "Deep Learning (Binary)":
            model = Sequential()
            model.add(Dense(64, activation='relu', input_dim=len(df.columns) - 1))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if st.button("🚀 Train Model"):
            if ml_type.startswith("Supervised") or ml_type.startswith("Deep"):
                X = df.drop(columns=[target_column])
                y = df[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                if ml_type.startswith("Supervised"):
                    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_test)

                    if ml_type == "Supervised - Classification":
                        st.success("✅ Classification Results")
                        st.text(classification_report(y_test, preds))
                        st.metric("Accuracy", round(accuracy_score(y_test, preds), 2))

                    elif ml_type == "Supervised - Regression":
                        st.success("✅ Regression Results")
                        st.metric("MSE", round(mean_squared_error(y_test, preds), 2))
                        st.metric("R2 Score", round(r2_score(y_test, preds), 2))

                elif ml_type == "Deep Learning (Binary)":
                    model.fit(X_train, y_train, epochs=10, verbose=0)
                    loss, acc = model.evaluate(X_test, y_test, verbose=0)
                    st.success("✅ Deep Learning Results")
                    st.metric("Accuracy", round(acc, 2))
                    st.metric("Loss", round(loss, 2))

            elif ml_type == "Unsupervised - Clustering":
                X = df.copy()
                model.fit(X)
                st.success("✅ Clustering Results")
                df['Cluster'] = model.labels_
                st.write(df[['Cluster']].value_counts())
                st.dataframe(df.head())

# Footer
st.markdown('<div class="coursemon-footer">Powered by Coursemon 🚀</div>', unsafe_allow_html=True)
