import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Employee Attrition Analysis"
)

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F8FAFC;
        background-image:
        radial-gradient(circle at 25% 90%, #DBEAFE 0%, transparent 80%),
        radial-gradient(circle at 90% 10%, #CCFBF1 0%, transparent 80%);
    }


    h1 {
    color: #1D4ED8;
    font-size: 42px;
    font-weight: 700;
    animation: fadeIn 2s ease-in;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

   h2, h3 {
    color: #2563EB;
    font-weight: 600;
    animation: slideRight 1s ease;
}

@keyframes slideRight {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
    

@keyframes slideRight {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}


    p, label {
        color: #334155;
    }


    .stDataFrame {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 10px;
    }


    .stAlert {
        border-radius: 12px;
    }


    div[data-testid="metric-container"] {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    transition: 0.3s;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}
.block-container {
    animation: slideUp 1s ease-in;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}


    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background-color:#FFFFFF;
        padding:25px;
        border-radius:15px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom:20px;
        animation: fadeIn 1.5s ease-in;
    ">

    <h1 style="color:#0F172A;">
    Employee Attrition Analysis
    </h1>

    <p style="color:#64748B;font-size:18px;">
    HR Analytics Dashboard using Machine Learning
    </p>

    </div>

    <style>
    @keyframes fadeIn {
        from {
            opacity:0;
            transform:translateY(-20px);
        }
        to {
            opacity:1;
            transform:translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    if os.path.exists("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"):
        return pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    else:
        return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

try:
    df = load_data()

except Exception as e:
    st.error(f"Dataset not found: {e}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Information")

st.write("Dataset Shape:")
st.write(df.shape)

st.write("Statistical Summary:")
st.dataframe(df.describe())

st.subheader("Missing Values")

st.dataframe(
    df.isnull().sum().to_frame("Missing Values")
)

st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(figsize=(3,2))

    df["Attrition"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Attrition Distribution",
        fontsize=9
    )

    ax.tick_params(
        labelsize=7
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

with col2:

    fig, ax = plt.subplots(figsize=(3,2))

    df["Gender"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Gender Distribution",
        fontsize=9
    )

    ax.tick_params(
        labelsize=7
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

col3, col4 = st.columns(2)

with col3:

    fig, ax = plt.subplots(figsize=(3,2))

    df["Department"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Department Distribution",
        fontsize=9
    )

    plt.xticks(
        rotation=20,
        fontsize=7
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

with col4:

    fig, ax = plt.subplots(figsize=(3,2))

    ax.hist(
        df["Age"],
        bins=20
    )

    ax.set_title(
        "Age Distribution",
        fontsize=9
    )

    ax.tick_params(
        labelsize=7
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

st.subheader("Data Preprocessing")

df1 = df.copy()
df1["Attrition"] = df1["Attrition"].map(
    {
        "Yes":1,
        "No":0
    }
)

drop_columns = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df1.drop(
    drop_columns,
    axis=1,
    inplace=True
)

for column in df1.select_dtypes(include="object").columns:

    le = LabelEncoder()

    df1[column] = le.fit_transform(
        df1[column]
    )

st.write("### Preprocessing Steps")

st.write("""
✅ Converted Attrition (Yes/No) into binary values  
✅ Removed unnecessary columns  
✅ Applied Label Encoding on categorical columns  
✅ Prepared dataset for Machine Learning model
""")

st.write("Processed Dataset Preview")

st.dataframe(df1.head())

X = df1.drop(
    "Attrition",
    axis=1
)

y = df1["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = GradientBoostingClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

st.subheader("Model Performance")
accuracy = accuracy_score(
    y_test,
    pred
)

st.success(
    f"Accuracy: {accuracy*100:.2f}%"
)

st.write("Confusion Matrix")
st.write(
    confusion_matrix(
        y_test,
        pred
    )
)

st.write("Classification Report")
st.text(
    classification_report(
        y_test,
        pred
    )
)

st.markdown(
    "### Conclusion"
)
st.write(
    "Gradient Boosting Classifier achieved approximately 89% accuracy on the IBM HR Analytics Employee Attrition dataset."
)
