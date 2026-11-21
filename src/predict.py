import os, sys, json
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # textprep.py must be importable to load the model
ROOT = os.path.dirname(HERE)

_model = joblib.load(os.path.join(ROOT, "models", "profanity_model.joblib"))
with open(os.path.join(ROOT, "models", "config.json"), encoding="utf-8") as f:
    _threshold = json.load(f)["threshold"]


def predict(text: str) -> dict:
    prob = float(_model.predict_proba([text])[0, 1])
    label = int(prob >= _threshold)
    return {"text": text, "label": label, "prob_profane": round(prob, 4),
            "meaning": "Kazakh profanity present" if label else "no Kazakh profanity"}


if __name__ == "__main__":
    texts = sys.argv[1:] or [input("Enter a Kazakh comment: ")]
    for t in texts:
        r = predict(t)
        print(f"{r['label']}  (P(profanity)={r['prob_profane']:.3f}, threshold={_threshold:.3f})  {r['meaning']}  <- {t}")
