# SHL Assessment Recommender — Approach Document

## Overview

A conversational FastAPI service that takes a hiring manager from vague intent to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

---

## Architecture

```
POST /chat (stateless, full history each call)
       │
       ▼
Retrieval (TF-IDF / SentenceTransformer + FAISS)
       │
       ▼
Context injection → Claude Haiku (Anthropic API)
       │
       ▼
JSON response: reply + recommendations[] + end_of_conversation
```

**Stack:** FastAPI + Anthropic Claude Haiku-4.5 + SentenceTransformer (all-MiniLM-L6-v2) + FAISS + scikit-learn (fallback). Deployed on Render (free tier).

---

## Catalog Construction

The SHL product catalog is JavaScript-rendered, making direct scraping from the container impractical. I scraped individual product detail pages (e.g. `/view/sql-new/`) via `web_fetch` to get structured data (name, description, test type, job levels, duration, languages), then compiled 68 Individual Test Solutions spanning all test types (A, B, C, D, K, P, S). Pre-packaged Job Solutions were excluded per the spec.

Each entry includes: name, canonical URL, description, test_type, job_levels, job_families, duration_minutes, remote_testing flag, languages, industries (where applicable), and a `keywords` array for retrieval enrichment.

---

## Retrieval Design

**Two-stage hybrid retrieval:**

1. **Dense (SentenceTransformer + FAISS):** `all-MiniLM-L6-v2` encodes documents as `name | description | job_levels | families | keywords` strings. Cosine similarity via FAISS IndexFlatIP. Activated at startup if HuggingFace is reachable.

2. **Sparse fallback (TF-IDF, bigrams):** When the model can't be loaded (cold-start without internet), a bigram TF-IDF with 8k features is used. This preserves technical term matching (e.g. "Java 8", "OPQ32r", "Selenium").

**Query construction:** Concatenates the last 6 user messages to capture refinement context. Top-20 results are injected into the system prompt for every turn.

**Why this works:** The catalog is small enough (68 items) that retrieving 20 of 68 with either method gives Claude strong coverage. The real intelligence—understanding role nuance, seniority, what to measure—lives in the LLM.

---

## Agent Design

**Model:** Claude Haiku 4.5 — fast enough to fit the 30-second timeout with retrieval, cheap enough for evaluation volume.

**Context engineering:** Every call includes:
- Role + behavior rules (scope enforcement, refusal patterns)
- Test type reference guide
- Role-to-assessment mapping hints (e.g. "IT developers → K+A+P")
- Top-20 retrieved catalog items with name, URL, type, description, levels
- JSON schema reminder injected at end of last user message

**Conversation behaviors:**
- **Clarify:** On vague queries, the agent asks ONE targeted question (role or what to measure). Forces clarification before first recommendation.
- **Recommend:** Once role + at least one dimension is known, produces 1–10 assessments.
- **Refine:** "Add personality" or "remove technical tests" → LLM updates the shortlist from catalog context.
- **Compare:** Answered from injected catalog data, not model prior → no hallucination risk.
- **Refuse:** Off-topic, legal, non-SHL queries get a short redirect.

**Anti-hallucination:** All recommendations pass through `validate_and_fix_recommendations()` which checks URLs and names against the catalog. Any item not matching a known URL or name is silently dropped.

**Turn cap compliance:** The LLM is instructed to make a best-effort recommendation after at most 2 clarifying turns, keeping total conversation ≤ 8 turns.

---

## Prompt Design

Output format is strict JSON enforced by:
1. System prompt specifying exact schema with examples
2. Per-turn reminder appended to last user message
3. Response parser that strips markdown fences and finds JSON via regex fallback

The catalog context is injected as a structured bullet list (not raw JSON) — easier for the model to reason over than a JSON blob.

---

## Evaluation Approach

**Local tests run:**
- Retrieval quality: verified top-5 results for "Java developer", "personality manager", "data analyst SQL" match expected assessments
- JSON schema compliance: tested parse_agent_response with well-formed and malformed inputs
- URL validation: confirmed all 68 catalog URLs are correctly indexed

**Behavior probes designed for:**
- Vague query → no recommendations on turn 1
- Off-topic → refusal with empty recommendations
- Refinement → updated shortlist without restarting
- Turn cap → recommendation by turn 3 at latest

**What didn't work / trade-offs:**
- Tried a more complex agentic loop with explicit tool calls for retrieval — added latency without quality gain vs. injecting catalog context directly
- Claude Sonnet gave better recommendations but was too slow for the 30s timeout on cold starts; Haiku was the right balance
- The catalog size (68 items) means some niche roles have limited coverage; this is a data quality issue, not an architecture one

---

## AI Tools Used

- Claude (this interface) for architecture planning, code generation, and iterative debugging
- All code was reviewed and understood before submission; design choices can be defended
