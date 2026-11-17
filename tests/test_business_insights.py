import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
from src.config import PROCESSED_DIR
from src.business_insights import run_business_insights


def test_mitigations_structure():
    ms = run_business_insights(verbose=False)
    for m in ms:
        assert "category" in m
        assert "severity" in m
        assert "evidence" in m
        assert "suggested_action" in m
        assert "caveat" in m   # every action has a caveat


def test_mitigations_json_saved():
    p = PROCESSED_DIR / "business_mitigations.json"
    assert p.exists()
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert "mitigations" in payload
    assert "scope_note" in payload
    assert "hypotheses" in payload["scope_note"].lower()


def test_mitigations_have_no_fraud_claims():
    """Every mitigation must have a caveat — the model never claims fraud."""
    ms = run_business_insights(verbose=False)
    for m in ms:
        assert m["caveat"] != "", f"Missing caveat on {m}"