# SHL Assessment Recommender — Approach Document

## Overview

A conversational FastAPI service that takes a hiring manager from vague intent to a grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue.

---

## Architecture

```
POST /chat (stateless, full history each call)
       │
       ▼
Retrieval (TF-IDF bigram, scikit-learn)
       │
       ▼
Context injection → Groq API (Llama 3.3 70B)
       │
       ▼
JSON response: reply + recommendations[] + end_of_conversation
```

**Stack:** FastAPI + Groq API (llama-3.3-70b-versatile) + scikit-learn TF-IDF. Deployed on Render (free tier).

---

## Catalog Construction

The SHL product catalog is JavaScript-rendered, making direct scraping impractical. Individual product detail pages (e.g. `/view/sql-new/`) were fetched to extract structured data, then compiled into 68 Individual Test Solutions spanning all test types (A, B, C, D, K, P, S). Pre-packaged Job Solutions were excluded per the spec.

Each entry includes: name, canonical URL, description, test_type, job_levels, job_families, duration_minutes, remote_testing flag, languages, industries (where applicable), and a `keywords` array for retrieval enrichment.

---

## Retrieval Design

**TF-IDF (bigram, 8k features):** Documents are encoded as `name | description | job_levels | families | keywords` strings. Query is built by concatenating the last 6 user messages to capture refinement context. Top-20 results are injected into the system prompt on every turn.

**Why this works:** The catalog is small (68 items), so retrieving 20 of 68 gives the LLM strong coverage. The real intelligence — understanding role nuance, seniority, what to measure — lives in the LLM. TF-IDF is reliable for technical term matching (e.g. "Java 8", "OPQ32r", "Selenium") and uses minimal memory, fitting within Render's free tier 512MB limit.

---

## Agent Design

**Model:** Groq API with `llama-3.3-70b-versatile` — fast (fits 30s timeout), free tier available, no region restrictions.

**Context engineering:** Every call includes:
- Role + behavior rules (scope enforcement, refusal patterns)
- Test type reference guide (A/B/C/D/K/P/S)
- Role-to-assessment mapping hints (e.g. "IT developers → K+A+P")
- Top-20 retrieved catalog items with name, URL, type, description, levels
- JSON schema reminder injected at end of last user message

**Conversation behaviors:**
- **Clarify:** On vague queries, asks ONE targeted question before recommending
- **Recommend:** Once role + dimension is known, produces 1–10 assessments
- **Refine:** Constraint updates → shortlist updated from catalog context
- **Compare:** Answered from injected catalog data only, not model prior
- **Refuse:** Off-topic, legal, non-SHL queries → empty recommendations array

**Anti-hallucination:** All recommendations pass through `validate_and_fix_recommendations()` which checks every URL and name against the catalog. Any item not matching a known entry is silently dropped.

**Turn cap compliance:** LLM instructed to recommend by turn 3 at latest, keeping total conversation ≤ 8 turns.

---

## Prompt Design

Strict JSON output enforced by:
1. System prompt specifying exact schema with field descriptions
2. Per-turn reminder appended to last user message
3. Response parser that strips markdown fences and finds JSON via regex fallback

Catalog context injected as structured bullet list — easier for the model to reason over than a raw JSON blob.

---

## Evaluation Approach

**Local tests run:**
- Retrieval quality: verified top-5 results for "Java developer", "personality manager", "data analyst SQL" match expected assessments
- JSON schema compliance: tested `parse_agent_response` with well-formed and malformed inputs
- URL validation: confirmed all 68 catalog URLs correctly indexed

**Behavior probes designed for:**
- Vague query → no recommendations on turn 1
- Off-topic → refusal with empty recommendations
- Refinement → updated shortlist without restarting
- Turn cap → recommendation by turn 3 at latest

**What didn't work / trade-offs:**
- `sentence-transformers` + FAISS exceeded Render free tier 512MB memory limit — removed in favour of TF-IDF which uses ~50MB
- A more complex agentic loop with explicit tool calls added latency without quality gain vs. injecting catalog context directly
- Catalog size (68 items) means some niche roles have limited coverage — a data quality issue, not an architecture one

---

## AI Tools Used

- Claude (Anthropic) for architecture planning, code generation, and iterative debugging
- All code was reviewed and understood before submission; design choices can be defended
