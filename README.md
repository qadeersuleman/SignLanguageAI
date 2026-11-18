# SignBridge AI — Sign Language to Screen & Voice Translator

A modern Python AI application inspired by **"How AI Understands Sign Language?"** and **"Common Hand Signs and Their Meanings"**. It captures live webcam gestures, detects 21-point hand landmarks via MediaPipe, classifies all 9 signs using a trained `scikit-learn` `RandomForestClassifier` (with two-hand & temporal wave detection), and displays the translation on screen **and** speaks it aloud via Voice!

## Supported Hand Signs (All 9 Poster Signs)
1. **Hello** — Open palm with fingers spread wide
2. **Thank You** — Thumbs up
3. **Yes** — OK sign (thumb & index circle, 3 fingers up)
4. **Stop** — Open palm with fingers pressed tightly together
5. **Go** — Index finger pointing horizontally sideways
6. **Peace** — Victory / V-sign (index & middle fingers up)
7. **Help** — Closed fist
8. **Please** — Both open palms pressed together (prayer / namaste pose)
9. **Good Bye** — Open hand waving side-to-side

## Features
- **Live 5-Step AI Pipeline Visualizer**: Lights up Steps 1 → 5 in real time (Camera Input → Hand Detection → AI Model → Prediction → Display & Voice).
- **Dual Voice Output**: Choose between **Browser Natural Voice** (Web Speech API with selectable voices, speed & pitch) or **Python Local TTS** (`spd-say` / `espeak` / `pyttsx3`).
- **Smart Sentence Builder**: Chains recognized signs into spoken sentences with Speak, Copy, Undo, and Clear controls.
- **Interactive 3×3 Poster Grid**: Highlights active detections in real time and lets you click any card to preview & speak it immediately.
- **Teach AI (Live Retraining)**: Record your own hand samples from the webcam and retrain `models/gesture_model.pkl` in one click.
- **Modern Glassmorphism UI**: Dark Aurora & Light themes, neon hand-skeleton overlay toggle, mirror toggle, and animated voice equalizer.

## How to Run (inside `venv`)
```bash
cd SignLanguageAI
source venv/bin/activate
python run.py
```
Then open `http://127.0.0.1:8000` in your browser (opens automatically by default).
