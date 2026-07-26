"""Dashboard Streamlit — Prédiction de matchs de foot (Projet_02).

Ce fichier réutilise exactement le pipeline du notebook `projet_02.ipynb` :
mêmes features, même modèle (RandomForestClassifier, n_estimators=200,
random_state=42), même standardisation. Le modèle est entraîné une seule
fois au démarrage (mis en cache par Streamlit), puis chaque prédiction est
quasi instantanée.

Pour lancer le dashboard :
    uv add streamlit
    uv run streamlit run app.py

(Exécute cette commande à la racine de Projet_02, à côté de utils.py et du
dossier data/.)
"""

import os

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils import FLAGS, add_recent_form, get_current_form, load_results, team_label

FEATURES = [
    "diff_avg_points",
    "diff_avg_goals_scored",
    "diff_avg_goals_conceded",
    "is_neutral",
    "is_friendly",
]


@st.cache_resource(show_spinner="Entraînement du modèle sur 30 ans de matchs...")
def train():
    """Reproduit le pipeline du notebook et renvoie tout ce qu'il faut pour prédire."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "results.csv")
    df = load_results(data_path)
    df = df[(df["date"] >= "1994-01-01") & (df["date"] <= "2026-06-30")].reset_index(drop=True)

    df = add_recent_form(df, window=10, min_matches=5)
    df = df.dropna(subset=["home_avg_points", "away_avg_points"]).reset_index(drop=True)

    data = df[df["home_score"] != df["away_score"]].copy()
    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    data["is_neutral"] = data["neutral"].astype(int)
    data["is_friendly"] = (data["tournament"] == "Friendly").astype(int)
    data["diff_avg_points"] = data["home_avg_points"] - data["away_avg_points"]
    data["diff_avg_goals_scored"] = data["home_avg_goals_scored"] - data["away_avg_goals_scored"]
    data["diff_avg_goals_conceded"] = data["home_avg_goals_conceded"] - data["away_avg_goals_conceded"]

    X = data[FEATURES]
    y = data["home_win"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return df, model, scaler, accuracy, teams


def predict_match(df, model, scaler, team_a, team_b):
    form_a = get_current_form(df, team_a)
    form_b = get_current_form(df, team_b)
    row = pd.DataFrame(
        [[
            form_a["avg_points"] - form_b["avg_points"],
            form_a["avg_goals_scored"] - form_b["avg_goals_scored"],
            form_a["avg_goals_conceded"] - form_b["avg_goals_conceded"],
            1,  # is_neutral : on simule un match sur terrain neutre
            0,  # is_friendly : on simule un match à enjeu
        ]],
        columns=FEATURES,
    )
    row_scaled = scaler.transform(row)
    proba_a_wins = model.predict_proba(row_scaled)[0][1]
    if proba_a_wins >= 0.5:
        return team_a, proba_a_wins
    return team_b, 1 - proba_a_wins


st.set_page_config(page_title="Qui gagnerait ce match ?", page_icon="⚽", layout="centered")

st.title("⚽ Qui gagnerait ce match ?")
st.caption(
    "Prédiction basée sur un Random Forest entraîné sur 30 ans de matchs internationaux "
    "(1994-2026) — Projet 02 du Cahier de Vacances Data & IA."
)

df, model, scaler, accuracy, teams = train()

st.info(f"Précision du modèle sur le jeu de test : **{accuracy:.1%}**", icon="🎯")

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Équipe A", teams, index=teams.index("France") if "France" in teams else 0)
with col2:
    default_b = "Brazil" if "Brazil" in teams else teams[1]
    team_b = st.selectbox("Équipe B", teams, index=teams.index(default_b) if default_b in teams else 1)

st.write("")

if team_a == team_b:
    st.warning("Choisis deux équipes différentes.")
else:
    if st.button("Prédire le résultat", type="primary", use_container_width=True):
        try:
            winner, probability = predict_match(df, model, scaler, team_a, team_b)
            st.write("")
            st.markdown(
                f"<h2 style='text-align:center'>{team_label(winner)} l'emporterait</h2>",
                unsafe_allow_html=True,
            )
            st.progress(probability, text=f"Probabilité de victoire : {probability:.0%}")
            st.write("")
            st.markdown(
                f"<p style='text-align:center;color:gray'>"
                f"{FLAGS.get(team_a,'🏳️')} {team_a}  vs  {FLAGS.get(team_b,'🏳️')} {team_b}"
                f"</p>",
                unsafe_allow_html=True,
            )
        except ValueError as e:
            st.error(str(e))

with st.expander("Comment ça marche ?"):
    st.markdown(
        """
Le modèle compare la **forme récente** des deux équipes (points, buts marqués et
encaissés en moyenne sur leurs 10 derniers matchs), et prédit qui l'emporterait sur
terrain neutre. Il ne connaît ni les blessures, ni la tactique, ni la forme du jour —
juste l'historique. Comme le rappelle le notebook : même les bookmakers professionnels
plafonnent autour de 70% sur ce genre de prédiction, le foot reste un sport à surprises !
        """
    )
