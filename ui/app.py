import streamlit as st
import requests
import json
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Asszisztens", layout="centered")
st.title("RAG alapú AI asszisztens")

st.write(
    "Tölts fel dokumentumokat (TXT/PDF), majd tegyél fel kérdéseket, "
    "és az asszisztens a dokumentumok alapján válaszol."
)

# session state inicializálás

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "history" not in st.session_state:
    st.session_state.history = [] 

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_context" not in st.session_state:
    st.session_state.last_context = []
if "last_monitoring" not in st.session_state:
    st.session_state.last_monitoring = {}
if "last_question" not in st.session_state:
    st.session_state.last_question = ""


# dokumentum feltöltése

st.subheader("Dokumentum feltöltése")

uploaded_file = st.file_uploader("Válassz egy TXT vagy PDF fájlt", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Feltöltés és indexelés"):
        with st.spinner("Feldolgozás és indexelés folyamatban..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Sikeres indexelés: {data['filename']} "
                        f"({data['chunks_indexed']} chunk került a vektortárba)"
                    )
                else:
                    st.error(f"Hiba ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Hiba a kérés során: {e}")

st.markdown("---")

# kérdés dokumentumok alapján

st.subheader("Kérdés dokumentumok alapján")

question = st.text_input("Írd be a kérdésed:")

col1, col2 = st.columns([1, 1])
with col1:
    send_clicked = st.button("Küldés")
with col2:
    if st.button("Új beszélgetés"):
        st.session_state.session_id = None
        st.session_state.history = []
        st.session_state.last_answer = None
        st.session_state.last_context = []
        st.session_state.last_monitoring = {}
        st.session_state.last_question = ""
        st.success("Új beszélgetés indítva")

# chat hívás

if send_clicked and question.strip():
    payload = {
        "question": question.strip(),
        "session_id": st.session_state.session_id,
    }

    with st.spinner("Válasz generálása..."):
        try:
            resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()

                st.session_state.session_id = data["session_id"]

                st.session_state.last_answer = data["answer"]
                st.session_state.last_context = data.get("context", [])
                st.session_state.last_monitoring = data.get("monitoring", {})
                st.session_state.last_question = question

                st.session_state.history.append({"role": "user", "content": question})
                st.session_state.history.append(
                    {"role": "assistant", "content": st.session_state.last_answer}
                )

            else:
                st.error(f"Hiba ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.error(f"Hiba a kérés során: {e}")


# válasz megjelenítése + kontextus + monitoring

if st.session_state.last_answer is not None:
    st.markdown("### ✅ Legutóbbi válasz")
    st.write(st.session_state.last_answer)

    with st.expander(" 🔍 Felhasznált kontextus (chunkok)"):
        if st.session_state.last_context:
            for i, c in enumerate(st.session_state.last_context, start=1):
                st.markdown(f"**Chunk #{i} - forrás: ** {c.get('source_file')}")
                st.write(c["text"][:500] + "...")
        else:
            st.write("Nem volt elérhető kontextus.")

    with st.expander(" Adatok monitorálása"):
        if st.session_state.last_monitoring:
            st.json(st.session_state.last_monitoring)
        else:
            st.write("Nincs elérhető monitoring információ.")

    # felhasználói feedback gyűjtés

    st.markdown("### ⭐ Felhasználói visszajelzés")
    rating = st.slider("Mennyire volt hasznos a válasz?", 1, 5, 4)
    comment = st.text_input("Megjegyzés (opcionális): ")

    if st.button("Visszajelzés küldése"):
        fb_path = Path("logs/feedback.jsonl")
        fb_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "session_id": st.session_state.session_id,
            "question": st.session_state.last_question,
            "answer": st.session_state.last_answer,
            "rating": rating,
            "comment": comment,
        }
        with fb_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success("Köszönöm a visszajelzést!")
