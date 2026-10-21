# Trustworthy AI Knowledge Assistant & Guardrail System

A complete, local, testable Python implementation of a **Trustworthy AI Knowledge Assistant** protected by a 9-stage input/output guardrail pipeline. Runs deterministically without paid APIs or external model dependencies while keeping the response generator decoupled (`ResponseGenerator` protocol) for future LLM integration.

---

## 1. Architecture & Six Core Guardrails

All six guardrails are implemented as standalone, testable functions in [`src/guardrails.py`](file:///home/himalbhandari/trustworthy-ai/src/guardrails.py) and orchestrated in strict order by [`TrustworthyKnowledgeAssistant`](file:///home/himalbhandari/trustworthy-ai/src/app.py#L237-L485) in [`src/app.py`](file:///home/himalbhandari/trustworthy-ai/src/app.py):

1. **Input PII Detection & Redaction (`detect_and_redact_pii`)**: Detects and redacts standard and obfuscated email addresses, telephone numbers, and configured identification numbers (`SSN`, `EMP_ID`, `CUST_ID`, `NATIONAL_ID`, `CREDIT_CARD`, plus custom regex patterns). Supports `REDACT` and `REFUSE` policies and masks/hashes findings so raw PII is never logged.
2. **Input Prompt-Injection Detection (`detect_prompt_injection`)**: Detects instruction overrides (`INSTRUCTION_OVERRIDE`), system prompt extraction (`SYSTEM_PROMPT_EXTRACTION`), restriction/jailbreak bypasses (`RESTRICTION_BYPASS`), and secret disclosure attempts (`SECRET_DISCLOSURE`).
3. **Input Sensitive-Topic Classification (`classify_sensitive_topic`)**: Enforces category policies across `SELF_HARM`, `WEAPONS_CBRN`, `CYBERATTACK_MALWARE`, `ILLEGAL_ACTS_FRAUD`, and `REGULATED_ADVICE`. Distinguishes legitimate educational/defensive inquiries (`ALLOW`) from actionable harmful requests (`REFUSE`), routing `SELF_HARM` crisis inputs to a 988 Lifeline safe response.
4. **Output Schema Validation (`validate_output_schema`)**: Enforces strict Pydantic JSON validation (`AssistantResponseSchema`) with required fields (`answer`, `citations`, `confidence`, `status`, `safety_notes`), permitted `Literal` values, length bounds, and valid citation ID checks.
5. **Output Groundedness Validation (`validate_groundedness`)**: Validates claims against trusted reference facts using citation existence checks, numeric/quantitative consistency, negation polarity & antonym contradiction detection, and content token grounding (`ALLOW` / `FALLBACK`).
6. **Output Sensitive-Information Detection (`detect_output_sensitive_info`)**: Scans generated outputs for PII, API keys/tokens, and configured runtime secrets, immediately rejecting leaked outputs (`REJECT` -> `FALLBACK`).

See [`docs/architecture.md`](file:///home/himalbhandari/trustworthy-ai/docs/architecture.md) for the 9-stage pipeline diagram and [`docs/evaluation_report.md`](file:///home/himalbhandari/trustworthy-ai/docs/evaluation_report.md) for full benchmark metrics and failure analysis.

---

## 2. Reproducible Setup & Execution Commands

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Application Demo
```bash
.venv/bin/python -m src.app
```

### Run a Single Query via CLI
```bash
# Standard human-readable output
.venv/bin/python -m src.app --query "What encryption standards are required for data at rest?"

# Full structured JSON pipeline output (includes stage-by-stage decisions & audit record)
.venv/bin/python -m src.app --query "My email is alice@example.com. When do authentication tokens expire?" --json
```

### Run the Automated Test Suite (`pytest`)
```bash
.venv/bin/pytest -v
```

### Run the Evaluation Benchmark Suite
```bash
.venv/bin/python -m src.evaluator
```

---

## 3. Usage Example (Python API)

```python
from src.app import TrustworthyKnowledgeAssistant

assistant = TrustworthyKnowledgeAssistant()

# 1. Query containing PII -> Redacted at Stage 2 and answered from DOC-102
result = assistant.handle_request(
    "Hi, my email is alice@company.org and ID is EMP-4821. When do authentication tokens expire?"
)
print(result.final_action.value)       # "REDACT"
print(result.sanitized_query)          # "Hi, my email is [REDACTED_EMAIL] and ID is [REDACTED_EMP_ID]. When do authentication tokens expire?"
print(result.response["status"])       # "ANSWERED"
print(result.response["citations"])    # ["DOC-102"]
print(result.response["answer"])       # "Authentication tokens and session credentials expire after 30 minutes of inactivity..."
```

---

## 4. Key Design Decisions & Limitations

### Design Decisions
- **Decoupled Generator Protocol**: [`ResponseGenerator`](file:///home/himalbhandari/trustworthy-ai/src/app.py#L81-L91) separates response generation from the 9-stage guardrail pipeline so an external LLM can replace [`DeterministicKnowledgeGenerator`](file:///home/himalbhandari/trustworthy-ai/src/app.py#L94-L150) without modifying any guardrail.
- **Untrusted Reference Isolation**: [`sanitize_untrusted_reference_facts`](file:///home/himalbhandari/trustworthy-ai/src/app.py#L157-L209) strips indirect prompt injections, PII, and secrets from reference documents before generation.
- **Bounded Retries vs. Non-Bypassable Security Checks**: Recoverable formatting/grounding errors (Stages 6 & 7) retry up to `max_retries`, whereas input security blocks (Stages 1–4) and output secret/PII leaks (Stage 8) halt immediately without retry.
- **Zero Raw PII/Secret Logging**: All findings and audit records mask (`ab***yz`) and SHA-256 hash sensitive strings.

### Documented Limitations
1. **Unstructured PII**: Regex detection covers structured emails, phone numbers, SSNs, and configured ID formats, but cannot identify free-form person names or unformatted street addresses without an NER model.
2. **Character-Obfuscated Prompt Injections (`INJ-05`)**: Syntactic regex patterns can be bypassed by leetspeak (`1gn0r3 a11 pr3v10us 1nstruct10ns...`), homoglyphs, or cipher encodings unless a character-normalization layer and semantic classifier are added.
3. **Abstractive Synonym Paraphrasing (`GRD-05`)**: The lexical/numeric groundedness validator catches numeric hallucinations and negation/antonym flips, but conservatively rejects heavy synonymous paraphrases (`"half an hour"` vs `"30 minutes"`, `"two-step verification"` vs `"MFA"`).

---

## 5. Actual Evaluation Results Summary

- **Unit & Integration Test Suite (`pytest -v`)**: **27 passed, 0 failed**
- **Evaluator Benchmark (`python -m src.evaluator`)**: **30 cases total — 28 passed, 2 concrete failure cases documented (`93.33%` accuracy)**

| Guardrail / Stage | Passed / Total | TP | TN | FP | FN | Precision | Recall | F1 Score | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1_input_pii` | 4 / 4 | 3 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `2_input_injection` | 4 / 5 | 3 | 1 | 0 | 1 | 1.0000 | 0.7500 | 0.8571 | 0.8000 |
| `3_sensitive_topic` | 5 / 5 | 3 | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `4_output_schema` | 4 / 4 | 3 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `5_output_groundedness` | 4 / 5 | 3 | 1 | 1 | 0 | 0.7500 | 1.0000 | 0.8571 | 0.8000 |
| `6_output_sensitive` | 3 / 3 | 2 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `7_end_to_end_pipeline` | 4 / 4 | 3 | 1 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Overall** | **28 / 30** | **20** | **8** | **1** | **1** | **0.9524** | **0.9524** | **0.9524** | **0.9333** |

- **Deployment Readiness Assessment**: **Suitable only with restrictions** (see [`docs/evaluation_report.md`](file:///home/himalbhandari/trustworthy-ai/docs/evaluation_report.md#L97-L103)).
