#!/usr/bin/env python3
"""
Local test script for SHL Assessment Recommender.
Run: python test_local.py
"""
import json
import requests
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    print("✓ /health OK")

def chat(messages):
    r = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=30
    )
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    return r.json()

def test_vague_query_no_recommendation():
    """Vague query should NOT produce recommendations on turn 1."""
    resp = chat([{"role": "user", "content": "I need an assessment"}])
    assert isinstance(resp["reply"], str) and len(resp["reply"]) > 10
    assert isinstance(resp["recommendations"], list)
    assert resp["recommendations"] == [], \
        f"Expected empty recs for vague query, got: {resp['recommendations']}"
    print("✓ Vague query → no recommendations on turn 1")

def test_java_developer():
    """Should recommend Java-related assessments."""
    msgs = [
        {"role": "user", "content": "I am hiring a mid-level Java developer who works with stakeholders"},
        {"role": "assistant", "content": json.dumps({"reply": "What specific skills would you like to assess?", "recommendations": [], "end_of_conversation": False})},
        {"role": "user", "content": "Technical Java skills and communication ability"},
    ]
    resp = chat(msgs)
    assert len(resp["recommendations"]) >= 1, "Should have at least 1 recommendation"
    names = [r["name"].lower() for r in resp["recommendations"]]
    has_java = any("java" in n for n in names)
    assert has_java, f"Expected Java-related assessment, got: {names}"
    # Validate URLs
    for rec in resp["recommendations"]:
        assert rec["url"].startswith("https://www.shl.com"), f"Bad URL: {rec['url']}"
        assert len(rec["test_type"]) == 1
    print(f"✓ Java developer → {len(resp['recommendations'])} recs including {[r['name'] for r in resp['recommendations']]}")

def test_off_topic_refused():
    """Off-topic questions should get empty recommendations."""
    resp = chat([{"role": "user", "content": "Can you give me advice on employment law in the UK?"}])
    assert resp["recommendations"] == [], \
        f"Off-topic should have empty recs, got: {resp['recommendations']}"
    print("✓ Off-topic → refused with empty recommendations")

def test_refinement():
    """Refinement should update the shortlist."""
    msgs = [
        {"role": "user", "content": "I want to hire a sales manager for a retail bank"},
        {"role": "assistant", "content": json.dumps({
            "reply": "Here are some assessments for a sales manager.",
            "recommendations": [
                {"name": "OPQ32r", "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/", "test_type": "P"},
                {"name": "Verify - Numerical Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-reasoning/", "test_type": "A"}
            ],
            "end_of_conversation": False
        })},
        {"role": "user", "content": "Also add a personality assessment that measures motivation"},
    ]
    resp = chat(msgs)
    assert len(resp["recommendations"]) >= 1, "Should have recommendations after refinement"
    print(f"✓ Refinement → {len(resp['recommendations'])} updated recs")

def test_schema_compliance():
    """All responses must match the schema."""
    resp = chat([{"role": "user", "content": "What assessments do you have for customer service?"}])
    assert "reply" in resp
    assert "recommendations" in resp
    assert "end_of_conversation" in resp
    assert isinstance(resp["reply"], str)
    assert isinstance(resp["recommendations"], list)
    assert isinstance(resp["end_of_conversation"], bool)
    assert len(resp["recommendations"]) <= 10
    print("✓ Schema compliance OK")

if __name__ == "__main__":
    print("Running SHL Recommender tests...\n")
    try:
        test_health()
        test_vague_query_no_recommendation()
        test_java_developer()
        test_off_topic_refused()
        test_refinement()
        test_schema_compliance()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Start it with: uvicorn main:app --reload")
        sys.exit(1)
