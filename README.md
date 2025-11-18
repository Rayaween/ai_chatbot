RAG alapú AI Asszisztens

Retrieval-Augmented Generation demonstrációs projekt

Ez a projekt egy teljesen működőképes, saját fejlesztésű RAG (Retrieval-Augmented Generation) rendszer, amely dokumentumok feldolgozását, releváns információk visszakeresését és LLM-alapú válaszgenerálást valósít meg.

A rendszer rendelkezik:
    - dokumentumfeltöltéssel
    - chunking + embedding + vektortárolással
    - OpenAI-alapú válaszgenerálással
    - Streamlit webes felülettel
    - háromszintű evaluation keretrendszerrel
    - monitoring funkciókkal

Videók: 
Technikai: https://www.loom.com/share/5f02800633a640c49551b7dad11e62ba
Felhasználói demó: https://www.loom.com/share/d6233cba2842409f9c1d5b2f93d07dc0

GitHub Repository link: https://github.com/Rayaween/ai_chatbot.git

File struktúra:
app/
  main.py            – FastAPI API (upload, chat, streaming)
  ingestion.py       – PDF/TXT feldolgozás, chunking
  embeddings.py      – OpenAI embedding
  vectordb.py        – Qdrant vektortár
  rag.py             – retrieval + reranking + válaszgenerálás
  monitoring.py      – token, költség, latency logolás

ui/
  app.py             – Streamlit UI
  monitoring_app.py  – Monitoring dashboard

eval/
  eval_retrieval.py
  eval_chunking.py
  eval_prompt.py
  eval_app.py

data/
  raw/               – feltöltött fájlok
  eval/              – tesztesetek

logs/
  requests.jsonl     – API logok
  feedback.jsonl     – UI feedback


1. Rendszer architektúra
A rendszer három fő komponensből áll:

A. Backend (FastAPI)
Funkciók:
- dokumentum feldolgozás (TXT/PDF)
- chunking stratégia (átfedéses, 500–700 token)
- embedding generálása
- vektoradatbázis építés
- releváns chunkok lekérdezése
- LLM válaszgenerálás
- token és időmérés, monitoring adatok visszaküldése

B. Vektoradatbázis

C. Frontend (Streamlit)
- dokumentum feltöltés
- kérdés → válasz interakció
- kontextus megjelenítés chunkokkal
- monitoring adatok megjelenítése
- user session kezelés
- feedback gyűjtés (rating + komment)


A projekt egy háromszintű értékelési rendszert tartalmaz, önálló scriptekkel.

1. RAG-szintű értékelés (20+ teszteset)
    Metrikák:
        - Precision: a visszaadott chunkok közül hány releváns
        - Recall: az összes relevánsból hányat talál meg
        - MRR (Mean Reciprocal Rank): a helyes chunk mennyire előkelő pozícióban jelent meg

    Futtatás:
    python -m eval.eval_retrieval
    python -m eval.eval_embeddings
    python -m eval.eval_chunking

2. Prompt-szintű értékelés (15+ teszteset)
    Teszteli:
        - a válasz helyességét
        - kontextushasználatot
        - hallucináció detektálást
        - LLM-as-Judge módszert
    Futtatás:
    python -m eval.eval_prompt

3. Alkalmazás-szintű értékelés (10 komplex teszt)
    Teszteli:
        - a teljes user journey-t
        - hibakezelést (üres fájl, nagy PDF, irreleváns kérdés stb.)
        - válaszlatenciát
        - session folytonosságot
    Futtatás:
    python -m eval.eval_app

4. Monitoring és analitika
    A válaszgenerálásnál a rendszer többek közt méri :
        - válaszideje
        - token szám
        - költség
        - chunkok száma
    Ezek Streamlit felületen megjelennek.

Telepítés és futtatás
    1. Klónozd
        git clone <https://github.com/Rayaween/ai_chatbot.git>
        cd ai_chatbot
    
    2. A gyökérkönyvtárban szereplő .env fileba illeszd be a használni kívánt OpenAI API kulcsot, majd mentsd el.

    2. Csomagok telepítése (requirements.txt)
        pip install -r requirements.txt

    3. Backend indítása
        uvicorn app.main:app --reload

    4. Frontend indítása
        streamlit run ui/app.py

Használat
    1. Nyisd meg a Streamlit UI-t.
    2. Tölts fel egy PDF vagy TXT dokumentumot.
    3. Várd meg a chunking + indexing visszaigazolást.
    4. Írj be egy kérdést, nyomj enter-t majd a küldés gombot és várt meg az asszisztens válaszát.
    5. Megjelenik:
        - a válasz
        - a felhasznált kontextus
        - monitoring adatok
        - értékelési felület, amin értékelhető a válasz minősége
    6. Indíts új beszélgetést vagy folytasd a meglévőben, ameddig van kérdésed.