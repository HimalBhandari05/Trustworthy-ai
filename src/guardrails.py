"""
Trustworthy AI Guardrail System — Core Guardrails Module.

Implements six distinct, testable safety guardrails:
1. Input PII detection and redaction (detect_and_redact_pii)
2. Input prompt-injection detection (detect_prompt_injection)
3. Input sensitive-topic classification and policy handling (classify_sensitive_topic)
4. Output Pydantic-based JSON schema validation (validate_output_schema)
5. Output groundedness validation against trusted reference facts (validate_groundedness)
6. Output sensitive-information (PII & secret) detection (detect_output_sensitive_info)

Security Guarantee:
- Never logs or exposes raw PII or secrets in findings (all findings are masked/hashed).
- Every guardrail returns structured decision information including category, action, and reason.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


# ============================================================================
# Action & Decision Data Models
# ============================================================================

class GuardrailAction(str, Enum):
    """Consistent guardrail action types across the input/output pipeline."""
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    REFUSE = "REFUSE"
    REJECT = "REJECT"
    FALLBACK = "FALLBACK"


@dataclass
class PIIFinding:
    """A single detected PII entity with masked preview (no raw PII stored)."""
    category: str
    masked_value: str
    value_hash: str
    start: int
    end: int


@dataclass
class GuardrailDecision:
    """Structured decision returned by every guardrail function."""
    guardrail_name: str
    passed: bool
    action: GuardrailAction
    category: str
    reason: str
    sanitized_text: Optional[str] = None
    findings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guardrail_name": self.guardrail_name,
            "passed": self.passed,
            "action": self.action.value,
            "category": self.category,
            "reason": self.reason,
            "findings": self.findings,
            "metadata": self.metadata,
        }


# ============================================================================
# Safe Masking & Hashing Helpers (Zero Raw PII / Secret Leakage)
# ============================================================================

def mask_sensitive_value(value: str) -> str:
    """Mask a sensitive string so raw PII or secrets are never logged or returned in findings."""
    clean = value.strip()
    if len(clean) <= 4:
        return "****"
    return f"{clean[:2]}***{clean[-2:]}"


def hash_sensitive_value(value: str) -> str:
    """Compute a short SHA-256 fingerprint of a sensitive value for safe audit correlation."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# Guardrail 1: Input PII Detection and Handling
# ============================================================================

DEFAULT_ID_PATTERNS: Dict[str, str] = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "EMP_ID": r"\bEMP-\d{4,6}\b",
    "CUST_ID": r"\bCUST-\d{5,8}\b",
    "NATIONAL_ID": r"\bNID-\d{6,10}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{16}\b",
}

CORE_PII_PATTERNS: Dict[str, str] = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "EMAIL_OBFUSCATED": r"\b[A-Za-z0-9._%+-]+\s*(?:\[at\]|\(at\)|\bat\b)\s*[A-Za-z0-9.-]+\s*(?:\[dot\]|\(dot\)|\bdot\b)\s*[A-Za-z]{2,}\b",
    "PHONE": r"(?:(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|\+\d{1,3}[\s.-]?\d{9,11}\b|\b\d{10}\b)",
}


def detect_and_redact_pii(
    text: str,
    custom_id_patterns: Optional[Dict[str, str]] = None,
    policy: Literal["REDACT", "REFUSE"] = "REDACT",
) -> GuardrailDecision:
    """
    Detect and redact PII in input text.

    Covers:
    - Common email addresses (including bracketed/word-obfuscated variants)
    - Telephone numbers (US/international formats)
    - Explicitly configured identification-number formats (SSN, EMP_ID, CUST_ID, etc.)

    Limitations:
    - Regex-based detection cannot identify unstructured context-dependent PII (e.g., full names
      without structural markers or physical street addresses without standard formatting).
    """
    patterns: Dict[str, str] = dict(CORE_PII_PATTERNS)
    id_configs = dict(DEFAULT_ID_PATTERNS)
    if custom_id_patterns:
        id_configs.update(custom_id_patterns)
    patterns.update(id_configs)

    # Collect all matches with spans
    raw_matches: List[Tuple[int, int, str, str]] = []
    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE if "OBFUSCATED" in pii_type else 0):
            start, end = match.span()
            raw_matches.append((start, end, pii_type, match.group(0)))

    if not raw_matches:
        return GuardrailDecision(
            guardrail_name="input_pii_guardrail",
            passed=True,
            action=GuardrailAction.ALLOW,
            category="NONE",
            reason="No PII detected in input.",
            sanitized_text=text,
            findings=[],
        )

    # Sort matches by start index and resolve overlapping spans (prefer longer match)
    raw_matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    deduped_matches: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for start, end, pii_type, val in raw_matches:
        if start >= last_end:
            deduped_matches.append((start, end, pii_type, val))
            last_end = end

    # Build sanitized text and safe findings
    findings: List[Dict[str, Any]] = []
    categories_found: List[str] = []
    sanitized_parts: List[str] = []
    cursor = 0

    for start, end, pii_type, val in deduped_matches:
        norm_type = "EMAIL" if pii_type == "EMAIL_OBFUSCATED" else pii_type
        if norm_type not in categories_found:
            categories_found.append(norm_type)

        findings.append({
            "type": norm_type,
            "masked_value": mask_sensitive_value(val),
            "value_hash": hash_sensitive_value(val),
            "span": [start, end],
        })
        sanitized_parts.append(text[cursor:start])
        sanitized_parts.append(f"[REDACTED_{norm_type}]")
        cursor = end

    sanitized_parts.append(text[cursor:])
    sanitized_text = "".join(sanitized_parts)
    category_summary = ",".join(categories_found)

    if policy == "REFUSE":
        return GuardrailDecision(
            guardrail_name="input_pii_guardrail",
            passed=False,
            action=GuardrailAction.REFUSE,
            category=category_summary,
            reason=f"Input refused because it contains sensitive PII ({category_summary}).",
            sanitized_text=sanitized_text,
            findings=findings,
        )

    return GuardrailDecision(
        guardrail_name="input_pii_guardrail",
        passed=True,
        action=GuardrailAction.REDACT,
        category=category_summary,
        reason=f"Detected and redacted {len(findings)} PII item(s) ({category_summary}).",
        sanitized_text=sanitized_text,
        findings=findings,
    )


# ============================================================================
# Guardrail 2: Input Prompt-Injection Detection
# ============================================================================

INJECTION_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "INSTRUCTION_OVERRIDE": [
        (
            r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|your\s+|these\s+|the\s+|any\s+)?"
            r"(?:previous|prior|above|earlier|system|initial|original|core)\s+"
            r"(?:instructions?|prompts?|directions?|rules?|constraints?|guidelines?|policies)",
            "Attempt to override prior/system instructions",
        ),
        (
            r"(?i)\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+(?:above|before|you\s+were\s+told|said\s+before)",
            "Attempt to clear instruction context",
        ),
        (
            r"(?i)(?:new|updated|revised|override)\s+(?:system\s+)?(?:instructions?|prompts?|directives?)\s*:",
            "Attempt to inject new system directive header",
        ),
        (
            r"(?i)\bdo\s+(?:exactly\s+)?what\s+i\s+(?:say|tell\s+you)\s+and\s+(?:not|ignore)\b",
            "Direct command override attempt",
        ),
        (
            r"(?i)(?:</?\s*system\s*>|\[\s*system\s*\]|###\s*system\s*instruction)",
            "Spoofed system delimiter tag",
        ),
    ],
    "SYSTEM_PROMPT_EXTRACTION": [
        (
            r"(?i)\b(?:print|output|reveal|show|display|dump|repeat|echo|give\s+me|tell\s+me)\s+"
            r"(?:me\s+)?(?:all\s+|everything\s+|the\s+|your\s+|full\s+|exact\s+|verbatim\s+)*"
            r"(?:system\s+|initial\s+|original\s+|hidden\s+|developer\s+|secret\s+)+"
            r"(?:prompts?|instructions?|messages?|rules?|directives?|guidelines?)",
            "Request to reveal system prompt or hidden instructions",
        ),
        (
            r"(?i)\bwhat\s+(?:are|were)\s+your\s+(?:exact\s+|original\s+|initial\s+|hidden\s+|system\s+)+"
            r"(?:instructions?|prompts?|rules?|directives?)",
            "Querying verbatim system instructions",
        ),
        (
            r"(?i)\brepeat\s+(?:the\s+words|everything|all\s+text)\s+(?:above|before\s+this|starting\s+with\s+['\"]?you\s+are)",
            "Verbatim prompt extraction via repetition trick",
        ),
    ],
    "RESTRICTION_BYPASS": [
        (
            r"(?i)\b(?:dan\s+mode|d\.a\.n\.|do\s+anything\s+now|jailbreak|developer\s+mode\s+enabled|god\s+mode)\b",
            "Known jailbreak persona trigger (DAN / Developer Mode)",
        ),
        (
            r"(?i)\byou\s+are\s+(?:now\s+)?(?:a\s+|an\s+)?(?:free|unconstrained|ungoverned|unrestricted|unfiltered|uncensored)\s+(?:ai|assistant|model)",
            "Persona reassignment to unrestricted AI",
        ),
        (
            r"(?i)\b(?:act\s+as\s+if|pretend|imagine|simulate)\s+(?:that\s+)?you\s+have\s+no\s+"
            r"(?:rules?|restrictions?|limits?|limitations?|ethics?|guardrails?|safety\s+filters?)",
            "Roleplay/hypothetical restriction bypass",
        ),
        (
            r"(?i)\b(?:bypass|disable|turn\s+off|deactivate|remove)\s+(?:all\s+|your\s+|the\s+)?"
            r"(?:safety\s+|security\s+|content\s+|ethical\s+)?(?:filters?|guardrails?|restrictions?|rules?|checks?|constraints?)",
            "Explicit request to disable safety guardrails",
        ),
    ],
    "SECRET_DISCLOSURE": [
        (
            r"(?i)\b(?:reveal|show|print|output|dump|give\s+me|extract|leak|disclose|tell\s+me|what\s+is|what\s+are)\s+"
            r"(?:the\s+|your\s+|all\s+|any\s+|internal\s+|configured\s+|hidden\s+|secret\s+)*"
            r"\b(?:api\s*keys?|secret\s*keys?|access\s*tokens?|private\s*keys?|passwords?|credentials?|env(?:ironment)?\s*variables?|database\s*urls?|canary\s*tokens?)\b"
            r"(?!\s+(?:storage|policy|policies|standard|standards|rules?|requirements?|expiration|rotation|guidelines?|management))",
            "Attempt to disclose secrets, API keys, or credentials",
        ),
        (
            r"(?i)\b(?:print|dump|read|cat|output)\s+(?:/etc/passwd|\.env|os\.environ|config\.secrets)",
            "Attempt to read system environment or credential files",
        ),
    ],
}


def detect_prompt_injection(text: str) -> GuardrailDecision:
    """
    Detect prompt injection attempts across four explicit threat categories:
    1. INSTRUCTION_OVERRIDE
    2. SYSTEM_PROMPT_EXTRACTION
    3. RESTRICTION_BYPASS
    4. SECRET_DISCLOSURE

    Limitations:
    - Pattern and heuristic detection can be evaded by novel semantic paraphrases,
      multi-language translations, or cipher encodings not covered by syntactic rules.
    """
    findings: List[Dict[str, Any]] = []
    matched_categories: List[str] = []

    for category, pattern_list in INJECTION_PATTERNS.items():
        for pattern, description in pattern_list:
            match = re.search(pattern, text)
            if match:
                if category not in matched_categories:
                    matched_categories.append(category)
                findings.append({
                    "category": category,
                    "description": description,
                    "matched_span": [match.start(), match.end()],
                })

    if findings:
        primary_category = matched_categories[0]
        all_categories = ",".join(matched_categories)
        return GuardrailDecision(
            guardrail_name="input_prompt_injection_guardrail",
            passed=False,
            action=GuardrailAction.REFUSE,
            category=primary_category,
            reason=f"Prompt injection detected ({all_categories}): {findings[0]['description']}.",
            findings=findings,
            metadata={"matched_categories": matched_categories},
        )

    return GuardrailDecision(
        guardrail_name="input_prompt_injection_guardrail",
        passed=True,
        action=GuardrailAction.ALLOW,
        category="NONE",
        reason="No prompt injection patterns detected.",
        findings=[],
    )


# ============================================================================
# Guardrail 3: Input Sensitive-Topic Classification & Policy Handling
# ============================================================================

# Benign / educational / defensive context indicators
LEGITIMATE_CONTEXT_MARKERS = [
    r"(?i)\b(?:what\s+is|what\s+are|explain|define|definition\s+of|history\s+of|overview\s+of)\b",
    r"(?i)\b(?:how\s+to\s+(?:prevent|protect|defend|mitigate|detect|remediate|stop|avoid|secure\s+against))\b",
    r"(?i)\b(?:defense\s+against|protection\s+from|mitigation\s+for|prevention\s+of|security\s+policy|compliance|awareness\s+training)\b",
    r"(?i)\b(?:why\s+is\s+.*\s+dangerous|impact\s+of|risks?\s+of|detecting\s+and\s+preventing)\b",
]

# Explicit harmful/actionable intent markers that override educational framing
HARMFUL_ACTION_MARKERS = [
    r"(?i)\b(?:how\s+(?:do\s+i|to|can\s+i)|step[\s-]by[\s-]step|instructions?\s+(?:for|on|to)|guide\s+to|tutorial\s+on)\s+"
    r"(?:make|build|create|synthesize|manufacture|write|execute|launch|commit|perform|hack|exploit|steal|launder)\b",
    r"(?i)\b(?:write|generate|give\s+me|provide)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:working\s+|functional\s+|undetectable\s+)?(?:exploit|payload|ransomware|keylogger|malware|phishing\s+email|bomb\s+recipe)\b",
]

SENSITIVE_TOPIC_POLICIES: Dict[str, Dict[str, Any]] = {
    "SELF_HARM": {
        "policy_name": "CRISIS_SAFE_REFUSAL",
        "restrictive_action": GuardrailAction.REFUSE,
        "safe_response_message": (
            "I cannot fulfill requests related to self-harm or suicide. "
            "If you or someone you know is going through a difficult time or is in crisis, "
            "please reach out for immediate support: call or text 988 (Suicide & Crisis Lifeline) "
            "or contact your local emergency services."
        ),
        "restrictive_patterns": [
            r"(?i)\b(?:i\s+(?:want|plan|am\s+going)\s+to\s+(?:kill\s+myself|end\s+my\s+life|commit\s+suicide|die|hurt\s+myself))\b",
            r"(?i)\b(?:how\s+to\s+(?:commit\s+suicide|kill\s+myself|harm\s+myself|overdose)|painless\s+suicide\s+methods?|ways\s+to\s+end\s+my\s+life)\b",
            r"(?i)\b(?:encourage|help|instruct)\s+(?:me|someone)\s+to\s+(?:self[\s-]?harm|commit\s+suicide)\b",
        ],
        "topic_keywords": [
            r"(?i)\b(?:suicide\s+prevention|self[\s-]?harm\s+awareness|mental\s+health\s+crisis\s+resources?)\b",
        ],
    },
    "WEAPONS_CBRN": {
        "policy_name": "ZERO_TOLERANCE_BLOCK",
        "restrictive_action": GuardrailAction.REFUSE,
        "safe_response_message": (
            "Request refused: I cannot provide instructions or assistance for creating weapons, "
            "explosives, or chemical, biological, radiological, or nuclear (CBRN) materials."
        ),
        "restrictive_patterns": [
            r"(?i)\b(?:make|build|construct|create|assemble|manufacture|synthesize)\b.*\b"
            r"(?:pipe\s+bomb|improvised\s+explosive|ied|bomb|detonator|ricin|sarin|anthrax|mustard\s+gas|VX\s+nerve\s+agent|chemical\s+weapon|biological\s+weapon|dirty\s+bomb)\b",
            r"(?i)\b(?:recipe|blueprint|instructions?|formula)\s+(?:for|to\s+make)\s+(?:a\s+)?(?:bomb|explosive|ricin|sarin|anthrax|napalm)\b",
        ],
        "topic_keywords": [
            r"(?i)\b(?:chemical\s+weapons?\s+convention|nuclear\s+non[\s-]?proliferation|history\s+of\s+cbrn|biological\s+weapons?\s+treaty)\b",
        ],
    },
    "CYBERATTACK_MALWARE": {
        "policy_name": "ACTIONABLE_EXPLOIT_BLOCK",
        "restrictive_action": GuardrailAction.REFUSE,
        "safe_response_message": (
            "Request refused: I cannot provide actionable exploit payloads, malware code, "
            "or instructions for unauthorized access. I can, however, discuss defensive security concepts and mitigation strategies."
        ),
        "restrictive_patterns": [
            r"(?i)\b(?:write|create|build|develop|generate|give\s+me)\b.*\b"
            r"(?:ransomware|keylogger|rootkit|trojan|botnet|credential\s+harvester|reverse\s+shell|exploit\s+payload|phishing\s+kit|sql\s+injection\s+payload)\b",
            r"(?i)\b(?:how\s+(?:to|do\s+i|can\s+i)\s+(?:hack\s+into|break\s+into|compromise|exfiltrate\s+data\s+from|ddos|brute[\s-]?force))\b",
        ],
        "topic_keywords": [
            r"(?i)\b(?:ransomware|malware|sql\s+injection|cross[\s-]?site\s+scripting|xss|phishing|zero[\s-]?day|ddos|buffer\s+overflow|cyberattack)\b",
        ],
    },
    "ILLEGAL_ACTS_FRAUD": {
        "policy_name": "ILLEGAL_ACTIVITY_BLOCK",
        "restrictive_action": GuardrailAction.REFUSE,
        "safe_response_message": (
            "Request refused: I cannot assist with financial fraud, money laundering, identity theft, "
            "or illegal substance synthesis."
        ),
        "restrictive_patterns": [
            r"(?i)\b(?:how\s+(?:to|do\s+i|can\s+i)|steps?\s+to|guide\s+to)\b.*\b"
            r"(?:launder\s+money|commit\s+(?:tax\s+|wire\s+|credit\s+card\s+)?fraud|forge\s+documents?|steal\s+identit(?:y|ies)|synthesize\s+(?:methamphetamine|fentanyl|illegal\s+drugs?)|evade\s+aml)\b",
            r"(?i)\b(?:create|generate)\s+(?:a\s+)?(?:fake\s+passport|counterfeit\s+invoice|ponzi\s+scheme\s+plan)\b",
        ],
        "topic_keywords": [
            r"(?i)\b(?:money\s+laundering|anti[\s-]?money\s+laundering|aml\s+compliance|fraud\s+detection|identity\s+theft\s+prevention)\b",
        ],
    },
    "REGULATED_ADVICE": {
        "policy_name": "REGULATED_PROFESSIONAL_DISCLAIMER",
        "restrictive_action": GuardrailAction.REFUSE,
        "safe_response_message": (
            "Request refused: I cannot provide personalized medical prescriptions, clinical diagnoses, "
            "or binding legal representation. Please consult a licensed medical or legal professional."
        ),
        "restrictive_patterns": [
            r"(?i)\b(?:prescribe\s+me|what\s+exact\s+dosage\s+of\s+.*\s+should\s+i\s+take|diagnose\s+my\s+chest\s+pain|draft\s+a\s+legally\s+binding\s+court\s+pleading\s+for\s+my\s+lawsuit)\b",
        ],
        "topic_keywords": [
            r"(?i)\b(?:hipaa|gdpr|medical\s+privacy|legal\s+compliance\s+framework)\b",
        ],
    },
}


def classify_sensitive_topic(text: str) -> GuardrailDecision:
    """
    Classify input for sensitive topics and apply category-specific policy handling.

    Distinguishes:
    - Actionable harmful requests (REFUSE with category-specific policy message, including crisis helpline for SELF_HARM).
    - Legitimate educational, defensive, or policy discussion of sensitive topics (ALLOW with context metadata).

    Limitations:
    - Intent disambiguation relies on linguistic markers for educational/defensive vs. actionable framing.
      Subtle dual-use requests phrased as academic inquiries may require deeper semantic analysis.
    """
    has_legitimate_framing = any(re.search(p, text) for p in LEGITIMATE_CONTEXT_MARKERS)
    has_harmful_intent = any(re.search(p, text) for p in HARMFUL_ACTION_MARKERS)

    # 1. Check restrictive patterns across all categories
    for category, config in SENSITIVE_TOPIC_POLICIES.items():
        for pattern in config["restrictive_patterns"]:
            match = re.search(pattern, text)
            if match:
                # Special check: if it matched a broad cyber/fraud pattern ONLY because of terms,
                # check if it's purely defensive ("how to prevent ransomware").
                # Note: SELF_HARM and WEAPONS_CBRN restrictive patterns are never overridden by educational framing.
                if (
                    category in ("CYBERATTACK_MALWARE", "ILLEGAL_ACTS_FRAUD")
                    and has_legitimate_framing
                    and not has_harmful_intent
                    and re.search(r"(?i)\b(?:prevent|defend|protect|mitigate|detect|stop|avoid)\b", text)
                ):
                    continue

                return GuardrailDecision(
                    guardrail_name="input_sensitive_topic_guardrail",
                    passed=False,
                    action=config["restrictive_action"],
                    category=category,
                    reason=f"Policy '{config['policy_name']}' triggered for category {category}.",
                    findings=[{
                        "category": category,
                        "policy": config["policy_name"],
                        "intent": "RESTRICTED_ACTIONABLE",
                        "matched_span": [match.start(), match.end()],
                    }],
                    metadata={
                        "policy_name": config["policy_name"],
                        "safe_response": config["safe_response_message"],
                        "intent": "RESTRICTED_ACTIONABLE",
                    },
                )

    # 2. Check if text discusses a sensitive topic in a legitimate/educational/defensive manner
    for category, config in SENSITIVE_TOPIC_POLICIES.items():
        for kw_pattern in config["topic_keywords"]:
            match = re.search(kw_pattern, text)
            if match:
                # If paired with explicit harmful action markers ("how do I execute a sql injection"), refuse!
                if has_harmful_intent and not re.search(r"(?i)\b(?:prevent|defend|protect|mitigate|detect|stop|avoid)\b", text):
                    return GuardrailDecision(
                        guardrail_name="input_sensitive_topic_guardrail",
                        passed=False,
                        action=config["restrictive_action"],
                        category=category,
                        reason=f"Policy '{config['policy_name']}' triggered due to actionable harmful intent on {category}.",
                        findings=[{
                            "category": category,
                            "policy": config["policy_name"],
                            "intent": "RESTRICTED_ACTIONABLE",
                            "matched_span": [match.start(), match.end()],
                        }],
                        metadata={
                            "policy_name": config["policy_name"],
                            "safe_response": config["safe_response_message"],
                            "intent": "RESTRICTED_ACTIONABLE",
                        },
                    )

                # Legitimate discussion allowed
                return GuardrailDecision(
                    guardrail_name="input_sensitive_topic_guardrail",
                    passed=True,
                    action=GuardrailAction.ALLOW,
                    category=f"BENIGN_{category}_DISCUSSION",
                    reason=(
                        f"Legitimate educational/defensive discussion of sensitive topic ({category}) "
                        f"permitted under policy '{config['policy_name']}'."
                    ),
                    findings=[{
                        "category": category,
                        "policy": config["policy_name"],
                        "intent": "LEGITIMATE_EDUCATIONAL_OR_DEFENSIVE",
                        "matched_span": [match.start(), match.end()],
                    }],
                    metadata={
                        "policy_name": config["policy_name"],
                        "intent": "LEGITIMATE_EDUCATIONAL_OR_DEFENSIVE",
                    },
                )

    return GuardrailDecision(
        guardrail_name="input_sensitive_topic_guardrail",
        passed=True,
        action=GuardrailAction.ALLOW,
        category="GENERAL",
        reason="No restricted sensitive topics detected.",
        findings=[],
        metadata={"intent": "GENERAL"},
    )


# ============================================================================
# Guardrail 4: Output Schema Validation (Pydantic-based)
# ============================================================================

class AssistantResponseSchema(BaseModel):
    """
    Strict Pydantic schema for the Trustworthy AI Knowledge Assistant output.
    Enforces required fields, field types, value bounds, and permitted enum values.
    """
    answer: str = Field(..., min_length=5, max_length=3000, description="Grounded response to the user query.")
    citations: List[str] = Field(..., description="List of trusted reference fact IDs supporting the answer.")
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(..., description="Confidence level in factual grounding.")
    status: Literal["ANSWERED", "INSUFFICIENT_CONTEXT", "REFUSED"] = Field(
        ..., description="Permitted status values."
    )
    safety_notes: List[str] = Field(default_factory=list, description="Optional safety or policy annotations.")

    @field_validator("answer")
    @classmethod
    def answer_not_whitespace(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("answer must not be empty or whitespace only.")
        return v.strip()

    @field_validator("citations")
    @classmethod
    def validate_citations_for_status(cls, v: List[str], info: Any) -> List[str]:
        for cid in v:
            if not isinstance(cid, str) or not re.match(r"^[A-Za-z0-9_-]+$", cid):
                raise ValueError(f"Invalid citation ID format: '{cid}'.")
        return v


def _extract_json_string(raw_output: str) -> Optional[str]:
    """Extract JSON object from raw text or markdown code fences."""
    stripped = raw_output.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    brace_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return None


def validate_output_schema(
    raw_output: str | Dict[str, Any],
    known_fact_ids: Optional[Sequence[str]] = None,
) -> Tuple[GuardrailDecision, Optional[AssistantResponseSchema]]:
    """
    Validate model output against AssistantResponseSchema using Pydantic.

    Enforces:
    - Valid JSON structure
    - Required fields (answer, citations, confidence, status)
    - Field types and permitted Literal values
    - Logical consistency: status=='ANSWERED' must include at least one valid citation ID

    Limitations:
    - Schema validation guarantees structural conformance and permitted enum values,
      but does not verify whether the cited facts actually support the prose in `answer`.
    """
    parsed_dict: Optional[Dict[str, Any]] = None

    if isinstance(raw_output, dict):
        parsed_dict = raw_output
    elif isinstance(raw_output, str):
        json_str = _extract_json_string(raw_output)
        if not json_str:
            return (
                GuardrailDecision(
                    guardrail_name="output_schema_guardrail",
                    passed=False,
                    action=GuardrailAction.REJECT,
                    category="MALFORMED_JSON",
                    reason="Output does not contain a valid JSON object.",
                    findings=[{"error": "No JSON object found in output"}],
                ),
                None,
            )
        try:
            loaded = json.loads(json_str)
            if not isinstance(loaded, dict):
                return (
                    GuardrailDecision(
                        guardrail_name="output_schema_guardrail",
                        passed=False,
                        action=GuardrailAction.REJECT,
                        category="INVALID_JSON_TYPE",
                        reason="Output JSON root must be an object/dictionary.",
                        findings=[{"error": f"Expected dict, got {type(loaded).__name__}"}],
                    ),
                    None,
                )
            parsed_dict = loaded
        except json.JSONDecodeError as exc:
            return (
                GuardrailDecision(
                    guardrail_name="output_schema_guardrail",
                    passed=False,
                    action=GuardrailAction.REJECT,
                    category="JSON_DECODE_ERROR",
                    reason=f"Invalid JSON syntax: {exc.msg}",
                    findings=[{"error": str(exc)}],
                ),
                None,
            )
    else:
        return (
            GuardrailDecision(
                guardrail_name="output_schema_guardrail",
                passed=False,
                action=GuardrailAction.REJECT,
                category="INVALID_INPUT_TYPE",
                reason="Output must be a JSON string or dict.",
                findings=[{"error": f"Unsupported type {type(raw_output).__name__}"}],
            ),
            None,
        )

    try:
        validated = AssistantResponseSchema.model_validate(parsed_dict)
    except ValidationError as exc:
        error_details = [
            {"loc": [str(x) for x in err["loc"]], "msg": err["msg"], "type": err["type"]}
            for err in exc.errors()
        ]
        return (
            GuardrailDecision(
                guardrail_name="output_schema_guardrail",
                passed=False,
                action=GuardrailAction.REJECT,
                category="SCHEMA_VALIDATION_ERROR",
                reason=f"Pydantic schema validation failed with {len(error_details)} error(s).",
                findings=error_details,
            ),
            None,
        )

    # Check domain-level schema constraints: ANSWERED requires non-empty citations
    if validated.status == "ANSWERED" and not validated.citations:
        return (
            GuardrailDecision(
                guardrail_name="output_schema_guardrail",
                passed=False,
                action=GuardrailAction.REJECT,
                category="MISSING_CITATIONS_FOR_ANSWER",
                reason="Responses with status 'ANSWERED' must include at least one reference citation.",
                findings=[{"loc": ["citations"], "msg": "Citations cannot be empty when status is ANSWERED"}],
            ),
            None,
        )

    if known_fact_ids is not None:
        unknown_ids = [cid for cid in validated.citations if cid not in known_fact_ids]
        if unknown_ids:
            return (
                GuardrailDecision(
                    guardrail_name="output_schema_guardrail",
                    passed=False,
                    action=GuardrailAction.REJECT,
                    category="NONEXISTENT_CITATION_ID",
                    reason=f"Output cited unknown reference ID(s): {unknown_ids}.",
                    findings=[{"unknown_citations": unknown_ids}],
                ),
                None,
            )

    return (
        GuardrailDecision(
            guardrail_name="output_schema_guardrail",
            passed=True,
            action=GuardrailAction.ALLOW,
            category="VALID_SCHEMA",
            reason="Output conforms to AssistantResponseSchema.",
            findings=[],
        ),
        validated,
    )


# ============================================================================
# Guardrail 5: Output Groundedness Validation
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "with", "by", "about", "of", "from",
    "as", "into", "like", "through", "after", "over", "between", "out",
    "against", "during", "without", "before", "under", "around", "among",
    "and", "or", "but", "if", "while", "because", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "has", "have", "had", "do", "does", "did",
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "cannot", "cant",
    "without", "prohibited", "forbidden", "disallowed", "disabled",
    "optional", "unnecessary", "false",
}

ANTONYM_PAIRS: List[Tuple[set[str], set[str]]] = [
    ({"required", "mandatory", "must", "enforced", "strict"}, {"optional", "voluntary", "unnecessary", "discretionary"}),
    ({"allowed", "permitted", "enabled", "supported", "authorized"}, {"prohibited", "forbidden", "banned", "disabled", "disallowed", "unauthorized"}),
    ({"encrypted", "hashed", "masked", "redacted", "anonymized"}, {"plaintext", "unencrypted", "unmasked", "raw", "exposed"}),
    ({"local", "internal", "on-premise", "offline"}, {"external", "public", "third-party", "cloud-only"}),
    ({"minimum", "at least"}, {"maximum", "at most"}),
    ({"always", "every", "all"}, {"never", "none"}),
]


def _extract_content_tokens(text: str) -> set[str]:
    """Extract lowercase alphanumeric content tokens excluding stopwords."""
    words = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _extract_numbers_and_units(text: str) -> List[str]:
    """Extract numeric tokens (integers, decimals, percentages, durations, versions) from text."""
    # Normalize written numbers for basic comparison
    norm_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "thirty": "30", "sixty": "60", "ninety": "90", "hundred": "100",
    }
    lowered = text.lower()
    for word, digit in norm_map.items():
        lowered = re.sub(rf"\b{word}\b", digit, lowered)
    # Extract numbers (ignoring citation-like IDs such as DOC-1 orKB-01)
    cleaned = re.sub(r"\b[a-z]+-\d+\b", "", lowered)
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", cleaned)


def _has_negation_or_antonym_conflict(claim: str, fact_text: str) -> Tuple[bool, str]:
    """
    Check if a claim contradicts a reference fact via polarity inversion (negation)
    or domain antonym substitution on shared subject terms.
    """
    claim_lower = claim.lower()
    fact_lower = fact_text.lower()

    claim_tokens = _extract_content_tokens(claim_lower)
    fact_tokens = _extract_content_tokens(fact_lower)
    shared_subject = (claim_tokens & fact_tokens) - NEGATION_WORDS

    if len(shared_subject) < 2:
        return False, ""

    # 1. Check explicit antonym pairs when subject terms overlap
    for pos_set, neg_set in ANTONYM_PAIRS:
        claim_has_pos = bool(claim_tokens & pos_set)
        claim_has_neg = bool(claim_tokens & neg_set)
        fact_has_pos = bool(fact_tokens & pos_set)
        fact_has_neg = bool(fact_tokens & neg_set)

        if (claim_has_pos and fact_has_neg and not fact_has_pos) or (
            claim_has_neg and fact_has_pos and not claim_has_neg == fact_has_neg
        ):
            conflict_terms = (claim_tokens & (pos_set | neg_set)) | (fact_tokens & (pos_set | neg_set))
            return True, f"Semantic antonym conflict on shared subject ({sorted(conflict_terms)})"

    # 2. Check negation polarity flip on high-overlap clauses
    claim_negations = set(re.findall(r"\b(?:not|never|no|cannot|can't|isn't|aren't|doesn't|don't|mustn't|shouldn't|without)\b", claim_lower))
    fact_negations = set(re.findall(r"\b(?:not|never|no|cannot|can't|isn't|aren't|doesn't|don't|mustn't|shouldn't|without)\b", fact_lower))

    if bool(claim_negations) != bool(fact_negations) and len(shared_subject) >= 3:
        return True, "Negation polarity mismatch between claim and reference fact"

    return False, ""


def _split_into_claims(answer: str) -> List[str]:
    """Split an answer into atomic sentence claims."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def validate_groundedness(
    response: AssistantResponseSchema,
    reference_facts: Dict[str, str],
    min_claim_support_ratio: float = 0.45,
) -> GuardrailDecision:
    """
    Validate that the response claims are factually grounded in the cited trusted reference facts.
    
    Multi-check verification (goes beyond simple keyword overlap):
    1. Citation validity & retrieval: Verifies cited reference IDs exist in `reference_facts`.
    2. Numeric/quantitative consistency: Verifies all numbers, percentages, and quantities in each
       claim appear in the cited reference facts (detects numeric hallucinations/contradictions).
    3. Polarity & antonym contradiction detection: Detects negation inversions (e.g. "is required"
       vs "is not required") and antonym flips (e.g. "encrypted" vs "plaintext", "allowed" vs "prohibited").
    4. Proposition token support: Verifies that key content tokens of every sentence claim are
       substantively grounded in the cited facts.

    Limitations:
    - Without a full Natural Language Inference (NLI) neural model, complex syntactic rephrasings,
      multi-hop reasoning across disparate sentences, or subtle implicatures may produce false
      positives or false negatives.
    """
    # If the model explicitly abstained or refused, groundedness check passes if no unsupported claims are made
    if response.status in ("INSUFFICIENT_CONTEXT", "REFUSED"):
        return GuardrailDecision(
            guardrail_name="output_groundedness_guardrail",
            passed=True,
            action=GuardrailAction.ALLOW,
            category="ABSTAINED_OR_REFUSED",
            reason=f"Response status is {response.status}; no factual claims require grounding.",
            findings=[],
        )

    # 1. Verify cited facts exist
    cited_facts: Dict[str, str] = {}
    missing_citations: List[str] = []
    for cid in response.citations:
        if cid in reference_facts:
            cited_facts[cid] = reference_facts[cid]
        else:
            missing_citations.append(cid)

    if missing_citations or not cited_facts:
        return GuardrailDecision(
            guardrail_name="output_groundedness_guardrail",
            passed=False,
            action=GuardrailAction.FALLBACK,
            category="UNVERIFIED_CITATION",
            reason=f"Response cites missing or untrusted reference IDs: {missing_citations}.",
            findings=[{"missing_citations": missing_citations}],
        )

    combined_facts_text = " ".join(cited_facts.values())
    combined_fact_tokens = _extract_content_tokens(combined_facts_text)
    combined_fact_numbers = set(_extract_numbers_and_units(combined_facts_text))

    claims = _split_into_claims(response.answer)
    if not claims:
        return GuardrailDecision(
            guardrail_name="output_groundedness_guardrail",
            passed=False,
            action=GuardrailAction.FALLBACK,
            category="EMPTY_CLAIMS",
            reason="No verifiable claims found in response answer.",
            findings=[],
        )

    findings: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []
    unsupported_claims: List[Dict[str, Any]] = []

    for claim in claims:
        claim_tokens = _extract_content_tokens(claim)
        if not claim_tokens:
            continue

        # Check A: Numeric / Quantitative consistency against cited facts
        claim_numbers = _extract_numbers_and_units(claim)
        unmatched_numbers = [num for num in claim_numbers if num not in combined_fact_numbers]
        if unmatched_numbers:
            contradiction_entry = {
                "claim": claim,
                "verdict": "CONTRADICTED_OR_UNSUPPORTED_NUMBER",
                "reason": f"Claim contains number(s) {unmatched_numbers} not present in cited facts {list(combined_fact_numbers)}.",
            }
            contradictions.append(contradiction_entry)
            findings.append(contradiction_entry)
            continue

        # Check B: Negation & Antonym Contradiction against each cited fact
        claim_contradicted = False
        for cid, fact_text in cited_facts.items():
            is_conflict, conflict_reason = _has_negation_or_antonym_conflict(claim, fact_text)
            if is_conflict:
                contradiction_entry = {
                    "claim": claim,
                    "fact_id": cid,
                    "verdict": "CONTRADICTED",
                    "reason": conflict_reason,
                }
                contradictions.append(contradiction_entry)
                findings.append(contradiction_entry)
                claim_contradicted = True
                break

        if claim_contradicted:
            continue

        # Check C: Content token support ratio
        overlap_tokens = claim_tokens & combined_fact_tokens
        support_ratio = len(overlap_tokens) / len(claim_tokens)

        if support_ratio < min_claim_support_ratio:
            unsupported_entry = {
                "claim": claim,
                "verdict": "UNSUPPORTED",
                "support_ratio": round(support_ratio, 2),
                "unmatched_tokens": sorted(claim_tokens - combined_fact_tokens),
                "reason": (
                    f"Claim token support ratio ({support_ratio:.2f}) is below threshold "
                    f"({min_claim_support_ratio:.2f})."
                ),
            }
            unsupported_claims.append(unsupported_entry)
            findings.append(unsupported_entry)
        else:
            findings.append({
                "claim": claim,
                "verdict": "SUPPORTED",
                "support_ratio": round(support_ratio, 2),
            })

    if contradictions:
        return GuardrailDecision(
            guardrail_name="output_groundedness_guardrail",
            passed=False,
            action=GuardrailAction.FALLBACK,
            category="FACTUAL_CONTRADICTION",
            reason=f"Detected {len(contradictions)} contradictory claim(s) against trusted reference facts.",
            findings=findings,
            metadata={"contradictions": len(contradictions), "unsupported": len(unsupported_claims)},
        )

    if unsupported_claims:
        return GuardrailDecision(
            guardrail_name="output_groundedness_guardrail",
            passed=False,
            action=GuardrailAction.FALLBACK,
            category="UNSUPPORTED_CLAIM",
            reason=f"Detected {len(unsupported_claims)} unsupported/unverified claim(s) not grounded in cited facts.",
            findings=findings,
            metadata={"contradictions": 0, "unsupported": len(unsupported_claims)},
        )

    return GuardrailDecision(
        guardrail_name="output_groundedness_guardrail",
        passed=True,
        action=GuardrailAction.ALLOW,
        category="GROUNDED",
        reason=f"All {len(claims)} claim(s) verified against cited reference facts.",
        findings=findings,
        metadata={"verified_claims": len(claims)},
    )


# ============================================================================
# Guardrail 6: Output Sensitive-Information Detection (PII & Configured Secrets)
# ============================================================================

DEFAULT_SECRET_PATTERNS: Dict[str, str] = {
    "OPENAI_API_KEY": r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b",
    "ANTHROPIC_API_KEY": r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
    "AWS_ACCESS_KEY": r"\bAKIA[0-9A-Z]{16}\b",
    "GITHUB_TOKEN": r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z_-]{35}\b",
    "JWT_TOKEN": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    "PRIVATE_KEY_BLOCK": r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
    "DB_CONNECTION_STRING": r"(?:postgresql|mongodb|mysql|redis)://[^:\s]+:[^@\s]+@[^\s]+",
    "GENERIC_SECRET_ASSIGNMENT": r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd|db[_-]?password)\s*[=:]\s*['\"]?[A-Za-z0-9_/+=-]{8,}['\"]?",
    "CANARY_TOKEN": r"\bCANARY-[A-F0-9]{8,16}\b",
}


def detect_output_sensitive_info(
    output_text: str,
    configured_secrets: Optional[Sequence[str]] = None,
    custom_secret_patterns: Optional[Dict[str, str]] = None,
    custom_id_patterns: Optional[Dict[str, str]] = None,
) -> GuardrailDecision:
    """
    Scan final output text for any leaked PII or configured secrets.

    Checks:
    1. Output PII via `detect_and_redact_pii` (emails, phones, SSNs, configured IDs).
    2. Pattern-based secret tokens (API keys, AWS keys, JWTs, private keys, DB URLs, canary tokens).
    3. Exact match against explicitly configured runtime secrets (`configured_secrets`).

    Security policy:
    - Any sensitive leak in model output causes immediate REJECT / FALLBACK so raw secrets or
      unredacted PII are never returned to the caller.
    - Raw secret values are never included in findings (masked and hashed only).

    Limitations:
    - Cannot detect novel secret values that neither match configured literal secrets nor
      conform to recognizable key/token patterns.
    """
    findings: List[Dict[str, Any]] = []
    categories: List[str] = []

    # 1. Check PII in output
    pii_decision = detect_and_redact_pii(output_text, custom_id_patterns=custom_id_patterns)
    if pii_decision.findings:
        categories.append("OUTPUT_PII")
        for f in pii_decision.findings:
            findings.append({
                "source": "PII_SCANNER",
                "type": f["type"],
                "masked_value": f["masked_value"],
                "value_hash": f["value_hash"],
            })

    # 2. Check regex secret patterns
    secret_patterns = dict(DEFAULT_SECRET_PATTERNS)
    if custom_secret_patterns:
        secret_patterns.update(custom_secret_patterns)

    for secret_type, pattern in secret_patterns.items():
        for match in re.finditer(pattern, output_text):
            val = match.group(0)
            if "OUTPUT_SECRET" not in categories:
                categories.append("OUTPUT_SECRET")
            findings.append({
                "source": "SECRET_PATTERN_SCANNER",
                "type": secret_type,
                "masked_value": mask_sensitive_value(val),
                "value_hash": hash_sensitive_value(val),
            })

    # 3. Check exact configured secrets
    if configured_secrets:
        for secret_val in configured_secrets:
            if secret_val and len(secret_val) >= 4 and secret_val in output_text:
                if "CONFIGURED_SECRET_LEAK" not in categories:
                    categories.append("CONFIGURED_SECRET_LEAK")
                findings.append({
                    "source": "CONFIGURED_SECRET_MATCH",
                    "type": "EXPLICIT_SECRET",
                    "masked_value": mask_sensitive_value(secret_val),
                    "value_hash": hash_sensitive_value(secret_val),
                })

    if findings:
        category_str = ",".join(categories)
        return GuardrailDecision(
            guardrail_name="output_sensitive_info_guardrail",
            passed=False,
            action=GuardrailAction.REJECT,
            category=category_str,
            reason=f"Blocked output containing sensitive information ({category_str}, {len(findings)} finding(s)).",
            findings=findings,
        )

    return GuardrailDecision(
        guardrail_name="output_sensitive_info_guardrail",
        passed=True,
        action=GuardrailAction.ALLOW,
        category="CLEAN",
        reason="No PII or secrets detected in output.",
        findings=[],
    )
