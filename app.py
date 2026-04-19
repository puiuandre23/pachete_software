import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


st.set_page_config(
    page_title="Analiza calitatii vinului",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.title-box {
    background-color: #f5f5f5;
    border: 1px solid #d9d9d9;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 20px;
}
.info-box {
    background-color: #fafafa;
    border-left: 4px solid #4f81bd;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.section-box {
    background-color: #fcfcfc;
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


def afiseaza_titlu():
    st.markdown("""
    <div class="title-box">
        <h1 style="margin-bottom:5px;">Proiect Pachete Software - Prima parte Python</h1>
        <p style="margin-top:0;">
            Analiza calitatii vinului folosind Streamlit, Pandas, Scikit-learn si Statsmodels
        </p>
    </div>
    """, unsafe_allow_html=True)


def afiseaza_obiectiv():
    st.markdown("""
    <div class="info-box">
        <b>Obiectiv:</b> analiza caracteristicilor chimice ale vinului si construirea unor modele
        care explica sau estimeaza nivelul calitatii.
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def incarca_date():
    df_local = pd.read_csv("WineQT.csv")
    return df_local


def trateaza_valori_lipsa(df):
    df_copy = df.copy()
    coloane_numerice = df_copy.select_dtypes(include=np.number).columns
    if df_copy[coloane_numerice].isnull().sum().sum() > 0:
        df_copy[coloane_numerice] = df_copy[coloane_numerice].fillna(
            df_copy[coloane_numerice].median()
        )
    return df_copy


def trateaza_outlieri(df, coloane):
    df_copy = df.copy()
    for col in coloane:
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        iqr = q3 - q1
        limita_inferioara = q1 - 1.5 * iqr
        limita_superioara = q3 + 1.5 * iqr
        df_copy[col] = np.where(df_copy[col] < limita_inferioara, limita_inferioara, df_copy[col])
        df_copy[col] = np.where(df_copy[col] > limita_superioara, limita_superioara, df_copy[col])
    return df_copy


def pregateste_date(df):
    df_copy = df.copy()

    if "Id" in df_copy.columns:
        df_copy = df_copy.drop(columns=["Id"])

    df_copy = trateaza_valori_lipsa(df_copy)

    coloane_numerice = df_copy.select_dtypes(include=np.number).columns.tolist()
    df_copy = trateaza_outlieri(df_copy, coloane_numerice)

    if "quality" in df_copy.columns:
        df_copy["quality_label"] = np.where(df_copy["quality"] >= 7, "Bun", "Slab")
        df_copy["quality_encoded"] = df_copy["quality_label"].map({"Slab": 0, "Bun": 1})

    return df_copy


def sectiune_incarcare(df_initial, df_final):
    st.header("1. Incarcarea datelor")

    st.success("Fisierul WineQT.csv a fost incarcat cu succes.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Numar randuri", df_final.shape[0])
    col2.metric("Numar coloane", df_final.shape[1])
    col3.metric("Valori lipsa initiale", int(df_initial.isnull().sum().sum()))

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Primele 5 inregistrari - head()")
    st.dataframe(df_final.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def sectiune_explorare(df):
    st.header("2. Explorare date")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Tipuri de date")
        tipuri = df.dtypes.astype(str).reset_index()
        tipuri.columns = ["Coloana", "Tip"]
        st.dataframe(tipuri, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Selectarea unei coloane")
        if "alcohol" in df.columns:
            st.write("Exemplu:")
            st.code("df['alcohol'].head()", language="python")
            st.dataframe(df["alcohol"].head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Statistici descriptive")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Exemplu loc / iloc")
        st.code("df.iloc[0:5, 0:4]", language="python")
        st.dataframe(df.iloc[0:5, 0:4], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def sectiune_valori_lipsa_si_extreme(df_initial, df_final):
    st.header("3. Valori lipsa si valori extreme")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Valori lipsa - isnull()")
        valori_lipsa = df_initial.isnull().sum().reset_index()
        valori_lipsa.columns = ["Coloana", "Numar valori lipsa"]
        st.dataframe(valori_lipsa, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Prelucrare aplicata")
        st.write("- valorile lipsa numerice au fost completate cu mediana")
        st.write("- valorile extreme au fost tratate prin metoda IQR")
        st.write("- s-a creat variabila categoriala quality_label")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Date dupa prelucrare")
    st.dataframe(df_final.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def sectiune_grupare(df):
    st.header("4. Grupare si agregare")

    if "quality" in df.columns:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Media indicatorilor pe niveluri de calitate")
        st.code("df.groupby('quality').mean(numeric_only=True)", language="python")
        grupare = df.groupby("quality").mean(numeric_only=True)
        st.dataframe(grupare, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Numarul observatiilor pe fiecare nivel de calitate")
        frecvente = df["quality"].value_counts().sort_index().reset_index()
        frecvente.columns = ["Calitate", "Numar observatii"]
        st.dataframe(frecvente, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def sectiune_grafice(df):
    st.header("5. Reprezentari grafice")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Distributia calitatii vinului")
        fig1, ax1 = plt.subplots()
        df["quality"].value_counts().sort_index().plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Calitate")
        ax1.set_ylabel("Numar observatii")
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Relatia dintre alcohol si quality")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df["alcohol"], df["quality"], alpha=0.6)
        ax2.set_xlabel("Alcohol")
        ax2.set_ylabel("Quality")
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Matrice de corelatie")
    corelatii = df.corr(numeric_only=True)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    cax = ax3.matshow(corelatii, cmap="coolwarm")
    plt.xticks(range(len(corelatii.columns)), corelatii.columns, rotation=90)
    plt.yticks(range(len(corelatii.columns)), corelatii.columns)
    fig3.colorbar(cax)
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)


def sectiune_clasificare(df):
    st.header("6. Clasificare - Regresie logistica")

    coloane_excluse = ["quality", "quality_label", "quality_encoded"]
    features = [col for col in df.columns if col not in coloane_excluse]

    X = df[features]
    y = df["quality_encoded"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acuratete = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Acuratete")
        st.metric("Accuracy", f"{acuratete:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Scalarea datelor")
        st.code("scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)", language="python")
        st.dataframe(pd.DataFrame(X_scaled, columns=features).head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.subheader("Matrice de confuzie")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Raport de clasificare")
    st.text(classification_report(y_test, y_pred))
    st.markdown('</div>', unsafe_allow_html=True)


def sectiune_regresie(df):
    st.header("7. Regresie multipla - Statsmodels")

    coloane_excluse = ["quality", "quality_label", "quality_encoded"]
    features = [col for col in df.columns if col not in coloane_excluse]

    y_reg = df["quality"]
    X_reg = df[features]
    X_reg = sm.add_constant(X_reg)

    model_reg = sm.OLS(y_reg, X_reg).fit()

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Rezumatul modelului")
    st.text(model_reg.summary())
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Interpretare")
    st.write("Modelul arata influenta variabilelor explicative asupra scorului de calitate.")
    st.write("Coeficientii pozitivi indica influente favorabile asupra calitatii.")
    st.write("Coeficientii negativi indica influente nefavorabile asupra calitatii.")
    st.markdown('</div>', unsafe_allow_html=True)



afiseaza_titlu()
afiseaza_obiectiv()

with st.sidebar:
    st.header("Navigare")
    sectiune = st.radio(
        "Alege sectiunea:",
        [
            "Incarcarea datelor",
            "Explorare date",
            "Valori lipsa si extreme",
            "Grupare si agregare",
            "Grafice",
            "Clasificare",
            "Regresie multipla",
        ]
    )

try:
    df_initial = incarca_date()
except FileNotFoundError:
    st.error("Fisierul WineQT.csv nu a fost gasit. Pune-l in acelasi folder cu app.py.")
    st.stop()

df_final = pregateste_date(df_initial)

if sectiune == "Incarcarea datelor":
    sectiune_incarcare(df_initial, df_final)
elif sectiune == "Explorare date":
    sectiune_explorare(df_final)
elif sectiune == "Valori lipsa si extreme":
    sectiune_valori_lipsa_si_extreme(df_initial, df_final)
elif sectiune == "Grupare si agregare":
    sectiune_grupare(df_final)
elif sectiune == "Grafice":
    sectiune_grafice(df_final)
elif sectiune == "Clasificare":
    sectiune_clasificare(df_final)
elif sectiune == "Regresie multipla":
    sectiune_regresie(df_final)
