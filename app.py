import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "loan_data 1.csv")

NUMERIC_FEATURES = [
    "credit.policy",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]
CATEGORICAL_FEATURE = "purpose"
TARGET = "not.fully.paid"

st.set_page_config(page_title="Decision Tree — Loan Default", page_icon="🌲", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_decision_tree(df: pd.DataFrame, max_depth, min_samples_leaf: int):
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=[CATEGORICAL_FEATURE], drop_first=True)
    y = df[TARGET]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(
        ascending=False
    )
    return model, feature_columns, acc, cm, importances


data_df = load_data()
purpose_options = sorted(data_df[CATEGORICAL_FEATURE].unique())

st.title("🌲 Will a borrower fail to fully pay their loan?")
st.caption("Decision tree classifier trained on `loan_data 1.csv`")

st.sidebar.header("Model settings")
max_depth_choice = st.sidebar.selectbox("Max tree depth", ["None", 2, 3, 4, 5, 6, 8, 10], index=4)
max_depth = None if max_depth_choice == "None" else int(max_depth_choice)
min_samples_leaf = st.sidebar.slider("Min samples per leaf", min_value=1, max_value=50, value=10)

model, feature_columns, acc, cm, importances = train_decision_tree(
    data_df, max_depth, min_samples_leaf
)

tab_predict, tab_explore, tab_data = st.tabs(["🔮 Predict", "📊 Explore", "🗂️ Data"])

with tab_predict:
    col_form, col_metrics = st.columns([2, 1])

    with col_form:
        st.subheader("Enter loan application details")
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_policy = st.selectbox("Meets credit policy", [1, 0], index=0)
            int_rate = st.number_input(
                "Interest rate", min_value=0.0, value=float(data_df["int.rate"].mean()), step=0.01
            )
            installment = st.number_input(
                "Installment", min_value=0.0, value=float(data_df["installment"].mean()), step=1.0
            )
            log_annual_inc = st.number_input(
                "Log annual income",
                min_value=0.0,
                value=float(data_df["log.annual.inc"].mean()),
                step=0.1,
            )
        with c2:
            dti = st.number_input(
                "Debt-to-income ratio", min_value=0.0, value=float(data_df["dti"].mean()), step=0.1
            )
            fico = st.number_input(
                "FICO score", min_value=0, value=int(data_df["fico"].mean()), step=1
            )
            days_with_cr_line = st.number_input(
                "Days with credit line",
                min_value=0.0,
                value=float(data_df["days.with.cr.line"].mean()),
                step=10.0,
            )
            revol_bal = st.number_input(
                "Revolving balance",
                min_value=0.0,
                value=float(data_df["revol.bal"].mean()),
                step=100.0,
            )
        with c3:
            revol_util = st.number_input(
                "Revolving utilization (%)",
                min_value=0.0,
                value=float(data_df["revol.util"].mean()),
                step=1.0,
            )
            inq_last_6mths = st.number_input(
                "Inquiries in last 6 months",
                min_value=0,
                value=int(data_df["inq.last.6mths"].mean()),
                step=1,
            )
            delinq_2yrs = st.number_input(
                "Delinquencies in last 2 years",
                min_value=0,
                value=int(data_df["delinq.2yrs"].mean()),
                step=1,
            )
            pub_rec = st.number_input(
                "Public derogatory records",
                min_value=0,
                value=int(data_df["pub.rec"].mean()),
                step=1,
            )

        purpose = st.selectbox("Loan purpose", purpose_options)

        if st.button("Predict Loan Status", type="primary"):
            raw_input = pd.DataFrame(
                [[
                    credit_policy, int_rate, installment, log_annual_inc, dti, fico,
                    days_with_cr_line, revol_bal, revol_util, inq_last_6mths, delinq_2yrs,
                    pub_rec, purpose,
                ]],
                columns=NUMERIC_FEATURES + [CATEGORICAL_FEATURE],
            )
            input_encoded = pd.get_dummies(raw_input, columns=[CATEGORICAL_FEATURE])
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0]
            classes = list(model.classes_)
            default_proba = proba[classes.index(1)]

            if prediction == 1:
                st.error(f"⚠️ Likely to NOT fully pay the loan (probability: {default_proba:.1%})")
            else:
                st.success(f"✅ Likely to fully pay the loan (probability: {1 - default_proba:.1%})")

    with col_metrics:
        st.subheader("Model performance")
        st.metric("Accuracy", f"{acc:.1%}")
        st.caption(f"Trained on {len(data_df):,} records")

        st.subheader("Confusion matrix")
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Fully Paid", "Not Fully Paid"],
            y=["Fully Paid", "Not Fully Paid"],
            color_continuous_scale="Oranges",
        )
        st.plotly_chart(cm_fig, width="stretch")

        st.subheader("Top feature importance")
        st.plotly_chart(
            px.bar(
                importances.head(10),
                orientation="h",
                labels={"index": "Feature", "value": "Importance"},
            ),
            width="stretch",
        )

with tab_explore:
    st.subheader("Feature vs Not Fully Paid")
    feature_choice = st.selectbox("Choose a numeric feature", NUMERIC_FEATURES, key="loan_feature")
    fig = px.box(
        data_df, x=TARGET, y=feature_choice, color=TARGET,
        title=f"{feature_choice} by {TARGET}",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Loan purpose distribution")
    st.plotly_chart(
        px.histogram(data_df, x=CATEGORICAL_FEATURE, color=TARGET, barmode="group"),
        width="stretch",
    )

with tab_data:
    st.subheader("Raw dataset")
    st.dataframe(data_df, width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(data_df.describe(), width="stretch")

