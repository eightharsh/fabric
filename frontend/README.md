# Frontend

A static, responsive web client (no build step) with a **minimal / technical**
design, split the way a product usually is:

| File | Role |
|---|---|
| `web/index.html` | **Landing page** — hero, pipeline schematic, how-it-works, "Launch app" |
| `web/app.html` | **Inspection console** — the actual tool |
| `web/app.js` | Console logic (health, upload/analyze, result rendering, view toggle) |
| `web/styles.css` | Shared design system + tokens (light + dark) |

The console uploads a fabric image to the backend `/predict` endpoint and shows
the **verdict**, stat tiles (anomaly score, threshold, regions, latency), and a
result view with an **Original / Heatmap / Boxes** toggle (defaults to Boxes on
the clean photo; boxes are drawn client-side from the returned coords so they
stay crisp). Provide an image by **Upload**, **Camera**, **Live camera**,
**drag-and-drop**, or **paste** (⌘/Ctrl+V). On phones the **Camera** button
opens the rear camera directly. The endpoint you set is remembered on the
device. Theme follows the OS by default with a manual light/dark toggle in the
nav; both are honoured across the app.

## Run it
It's a static file — no build step.

```bash
# from the project root, with the backend already running on :8000
cd frontend/web
python3 -m http.server 5173
# open http://localhost:5173  (desktop)
```

### Use from a phone
1. Put the phone and computer on the same Wi-Fi.
2. Find your computer's LAN IP (`ipconfig getifaddr en0` on macOS).
3. Serve the backend on `0.0.0.0` (already the default in `backend/main.py`).
4. On the phone, open `http://<computer-ip>:5173` and set the endpoint field to
   `http://<computer-ip>:8000/predict`.
5. For a Colab-hosted model, expose it with ngrok and paste that URL instead.

## Making it a real installable app (optional, later)
- **PWA**: add a `manifest.json` + service worker → "Add to Home Screen".
- **Native wrapper**: [Capacitor](https://capacitorjs.com/) wraps this same
  HTML/JS into an Android/iOS app with minimal changes.
- **From scratch**: rebuild the UI in React Native or Flutter hitting the same
  `/predict` API — the backend contract does not change.

## Note on coordinates
`boxes` are in the resized inference frame (`image_size`, default 224px). To draw
them on the original photo, scale by `original_dim / image_size`.
