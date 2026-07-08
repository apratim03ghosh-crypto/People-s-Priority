# People's Priorities — AI for Constituency Development Planning

A working end-to-end prototype for the *Build with AI: Code for Communities* hackathon.
Citizens submit development requests (text / voice / photo, any language) → Gemini turns
each one into structured data → those are clustered into themes and blended with
demographic/infrastructure data → the MP dashboard shows a ranked, AI-justified priority list
on a hotspot map.

## Architecture

```
frontend/index.html          Citizen intake form + MP dashboard (vanilla JS, no build step)
backend/main.py               FastAPI app: /api/submissions, /api/priorities, /api/map-data
backend/ai_engine.py          Gemini multimodal analysis + clustering/ranking logic
backend/data/demographics_sample.csv   Mock public dataset (replace with real Census/data.gov.in pulls)
backend/data/submissions.json Local JSON store created at runtime (swap for Firestore — see below)
```

**Why this shape:** Gemini 2.5's native multimodal + multilingual understanding means one
model call (text + photo + audio together) replaces a Speech-to-Text → Translation →
classification pipeline, which is the simplest path to a *working end-to-end flow* in
hackathon time. The demographic-weighting layer is a small, explainable rule set (see
`CATEGORY_DEMOGRAPHIC_RULES` in `ai_engine.py`) rather than a black box, so an MP's office
can see *why* something ranked where it did — then Gemini writes the human-readable
rationale on top of those numbers, grounded in the data, not invented.

## Quick start (local)

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your GEMINI_API_KEY into .env
python main.py               # serves on http://localhost:8080
```

Open `frontend/index.html` directly in a browser (or serve it: `python3 -m http.server 5500`
from inside `frontend/`). Edit `frontend/config.js` if your backend isn't on
`localhost:8080`, or if you have a Google Maps JavaScript API key to enable the hotspot map.

Get a Gemini API key at https://aistudio.google.com/apikey (free tier is enough for a demo).

## Try it

1. Open the **Submit a Request** tab, type or speak a complaint (try a non-English language),
   optionally attach a photo, click "Use my location", and submit.
2. Submit 4-5 varied requests (e.g. a few about roads, a couple about schools) so the
   ranking has something to cluster.
3. Switch to **MP Dashboard** — you'll see a ranked list with composite scores, affected
   wards, and an AI-written rationale + recommended action for each priority.

## Going from prototype to production

- **Storage:** `StorageBackend` in `main.py` is a single-file JSON store on purpose — swap
  its three methods for Firestore (`db.collection("submissions").add(record)` /
  `.stream()`) and nothing else in the app needs to change.
- **Voice/SMS at scale:** for low-connectivity citizens, front this API with the WhatsApp
  Business API or an SMS gateway — same `/api/submissions` payload shape, just populate
  `channel` accordingly.
- **Real demographic data:** `demographics_sample.csv` is illustrative. Swap in actual
  ward-level Census/NFHS/data.gov.in extracts, or move the join into BigQuery and query it
  from `build_priorities()` instead of reading a CSV.
- **Deploy the backend to Cloud Run:**
  ```bash
  cd backend
  gcloud run deploy peoples-priorities-api \
    --source . \
    --region asia-south1 \
    --set-env-vars GEMINI_API_KEY=your_key_here \
    --allow-unauthenticated
  ```
  Then point `frontend/config.js`'s `PEOPLES_PRIORITIES_API_BASE` at the resulting URL.
- **Host the frontend on Firebase Hosting:**
  ```bash
  npm install -g firebase-tools
  firebase init hosting   # set frontend/ as the public dir
  firebase deploy
  ```
- **Switching to Vertex AI auth instead of an API key:** replace
  `genai.Client(api_key=...)` in `ai_engine.py` with
  `genai.Client(vertexai=True, project="your-project", location="asia-south1")` and run
  `gcloud auth application-default login` (or use the Cloud Run service account — no key
  needed at all in that case).

## Data & API sources used

- Gemini API (`google-genai` SDK) for multimodal/multilingual submission analysis and
  rationale generation.
- Demographic/infrastructure figures in `demographics_sample.csv` are illustrative mock
  values modeled on the kind of fields available from Census of India / NFHS / data.gov.in
  ward-level datasets — replace with real extracts before any real deployment.
- Google Maps JavaScript API for the hotspot map (optional; dashboard works without it).

## License / originality note

All code here is original, written for this submission. Third-party dependencies
(`fastapi`, `google-genai`, `uvicorn`, etc.) are open-source packages used via their
published APIs, listed in `backend/requirements.txt`.
