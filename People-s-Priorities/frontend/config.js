// ── Deployment configuration ────────────────────────────────────────────────
//
// LOCAL:   Leave API_BASE as "" — FastAPI serves index.html so all /api/*
//          calls go to the same server automatically.
//
// NETLIFY: After your first Render deploy, paste your Render URL below, e.g.:
//          window.PEOPLES_PRIORITIES_API_BASE = "https://peoples-priority-api.onrender.com";
//
window.PEOPLES_PRIORITIES_API_BASE = "";

// Optional: Google Maps JavaScript API key for the hotspot heatmap.
// Leave blank — the ranked priority list works fine without the map.
window.PEOPLES_PRIORITIES_MAPS_KEY = "";
