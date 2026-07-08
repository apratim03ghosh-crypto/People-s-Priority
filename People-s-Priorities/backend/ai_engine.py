from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOCAL_DEMOGRAPHICS_CSV = BASE_DIR / "data" / "demographics_sample.csv"

Category = Literal[
    "roads", "water", "sanitation", "health", "education",
    "safety", "electricity", "housing", "environment", "other",
]

TriageCategory = Literal["quick_fix", "urgent_infrastructure", "long_term_planning", "critical_emergency"]


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class GeoPoint(BaseModel):
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lng: Optional[float] = Field(default=None, ge=-180, le=180)


class FormalizationResult(BaseModel):
    """
    Step 1 output: a cleaned, translated, profanity-free, slang-free version
    of the raw citizen complaint, suitable for a government record.
    """
    formal_description: str = Field(
        description=(
            "A professional, formal English translation and description of the civic issue. "
            "You MUST translate everything into standard, formal English. "
            "ALL profanity (e.g., 'sala', 'ch**iya') and offensive language MUST be completely stripped and replaced with "
            "neutral factual language. ALL slang, colloquialisms, abbreviations, and "
            "emotional overstatements must be converted to standard English. "
            "The factual civic complaint — location, problem type, severity — must be fully preserved."
        )
    )
    detected_language: str = Field(
        description=(
            "ISO 639-1 language code of the primary language used in the original submission. "
            "Examples: 'en' (English), 'hi' (Hindi), 'bn' (Bengali), 'ta' (Tamil), "
            "'te' (Telugu), 'mr' (Marathi), 'pa' (Punjabi), 'ml' (Malayalam), "
            "'gu' (Gujarati), 'kn' (Kannada), 'ur' (Urdu), 'or' (Odia)."
        )
    )
    profanity_detected: bool = Field(
        description=(
            "Set to true if the original text contained ANY profanity, swear words, "
            "derogatory terms, or offensive language in any language or script."
        )
    )
    cleaned_summary: str = Field(
        description=(
            "A single clear sentence summarizing the civic issue in professional English, "
            "suitable as a headline for a government record."
        )
    )


class DeduplicationResult(BaseModel):
    """
    Step 2 output: whether this new submission is a duplicate of an existing active issue.
    Only mark as duplicate when you are highly confident (>= 0.85).
    """
    is_duplicate: bool = Field(
        description=(
            "True ONLY if the new submission describes the EXACT SAME real-world problem "
            "at the SAME physical location as an existing active issue. "
            "Set to false if the location differs, the problem type differs, or you are uncertain. "
            "When in doubt, set is_duplicate=false to avoid false merges."
        )
    )
    master_id: Optional[str] = Field(
        default=None,
        description=(
            "The ID of the matching existing master issue. "
            "Required (non-null) when is_duplicate is true. Must be null when is_duplicate is false."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence in this decision, from 0.0 (completely uncertain) to 1.0 (certain). "
            "Only set is_duplicate=true if confidence >= 0.85."
        ),
    )
    reasoning: str = Field(
        description="A brief, specific explanation of why this is or is not a duplicate.",
    )


class SubmissionAnalysis(BaseModel):
    """Step 3 output: full structured civic issue analysis for a unique new issue."""

    summary: str = Field(description="One sentence citizen-friendly issue summary.")
    category: Category
    triage_category: TriageCategory = Field(
        description=(
            "Classify into exactly one of FOUR triage buckets (read all before deciding):\n"
            "• critical_emergency — IMMEDIATE danger to human life or public safety. "
            "Always assign urgency_score 9 or 10. Requires emergency services, police, or "
            "immediate district-level escalation. "
            "Examples: reported dead body, serious accident with injuries, building collapse, "
            "fire, gas leak, explosion, people trapped, drowning, murder scene.\n"
            "• urgent_infrastructure — Safety-critical or health-critical, needs same-day / "
            "next-day escalation. Typically urgency 7–8. "
            "Examples: bridge structural damage, hospital/clinic failure, major power outage, "
            "sewage overflow on streets, severe flooding, water contamination.\n"
            "• quick_fix — Minor, fast-resolvable issues needing routine maintenance. "
            "Typically urgency 3–6. "
            "Examples: potholes, broken streetlights, garbage collection, broken benches, "
            "minor drainage blocks, missing road signs, damaged footpaths.\n"
            "• long_term_planning — Issues requiring formal planning, budget approval, "
            "and multi-month execution. Typically urgency 1–5. "
            "Examples: building new roads, schools, parks, community centers, "
            "major drainage networks, new highways, public transit."
        )
    )
    urgency_score: float = Field(
        ge=1.0, le=10.0,
        description=(
            "MANDATORY: Assign a highly specific decimal float score (e.g., 3.4, 4.7, 7.2, 8.9). "
            "INTEGER SCORES like 4.0, 5.0, or 6.0 are STRICTLY FORBIDDEN unless mathematically exact. "
            "Calculate using deep granularity: public safety impact, population disruption, systemic scale. "
            "NEVER default to 5.0. Use the calibration benchmarks above to anchor your reasoning."
        ),
    )
    sentiment: Literal["negative", "neutral", "positive"]
    suggested_department: str
    constituency_priority: str = Field(
        description="Why this issue matters for the constituency development plan."
    )
    keywords: List[str] = Field(default_factory=list, max_length=8)
    location_hint: Optional[str] = Field(
        default=None,
        description="Named place or landmark mentioned by the citizen. Leave null if not mentioned.",
    )
    geocode: Optional[GeoPoint] = None
    rationale: str = Field(
        description="Brief evidence for the category, triage_category, and urgency_score assignments."
    )


# ─── System Prompts ───────────────────────────────────────────────────────────

FORMALIZATION_SYSTEM = """\
You are a civic issue formalization engine for an Indian constituency management platform.

Your SOLE job is to convert raw citizen complaints into clean, formal, government-record-ready English.

The raw input may contain ANY combination of the following — handle ALL of them:

═══ LANGUAGE ISSUES ════════════════════════════════════════════════════════════
• Regional Indian languages in native script:
  Hindi (Devanagari), Bengali, Tamil, Telugu, Marathi, Punjabi (Gurmukhi),
  Malayalam, Gujarati, Odia, Kannada, Assamese, Urdu, Rajasthani, Bhojpuri, etc.
• Transliteration and code-switching:
  Hinglish, Banglish, Tanglish, Manglish, mixed-script text, Roman-script regional languages
• Mixed multilingual sentences (e.g., English structure with Hindi words)
→ TRANSLATE ALL non-English content to formal English.

═══ LANGUAGE QUALITY ISSUES ════════════════════════════════════════════════════
• Slang, colloquial expressions, street language (e.g., "jugaad", "bakwaas", "ekdum kharab")
• Informal abbreviations (ASAP, FYI, etc. in unprofessional context)
• Emotional overstatements ("This road is literally killing us!", "Worst thing ever!")
• Repetition and filler words ("yaar", "bhai", "abey", etc.)
→ Convert to factual, professional, civic complaint language suitable for government records.

═══ PROFANITY AND OFFENSIVE CONTENT ════════════════════════════════════════════
• Swear words and profanity in ANY language or script
• Derogatory or abusive terms directed at officials, groups, or infrastructure
• Vulgar or sexually explicit language
• Hate speech or discriminatory content
→ REMOVE completely. Replace with neutral factual descriptions of the civic issue.

EXAMPLES OF PROFANITY HANDLING:
  Input:  "Ye haramkhor neta ka road bilkul ch**iya bana rakha hai"
  Output: "The road in this area has been severely neglected and requires urgent repair."

  Input:  "This f***ing pothole broke my bike yesterday near the market!!"
  Output: "A large pothole near the local market caused vehicle damage and requires immediate repair."

  Input:  "ভাঙা রাস্তা, শালা সরকার কিচ্ছু করে না, গাড়ি চলতেই পারে না"
  Output: "The road is severely damaged, preventing vehicles from passing. Government action is urgently needed."

═══ PRESERVATION RULES ══════════════════════════════════════════════════════════
• ALWAYS preserve the factual civic details: location, problem type, severity, impact
• DO NOT add information not present in the original
• DO NOT minimize or exaggerate the severity
• Set profanity_detected=true if ANY profanity or offensive language was present in the original
• Detect and report the primary language code used

Output ONLY the structured JSON. No preamble, no explanation.
"""

ANALYSIS_SYSTEM = """\
You are an AI assistant for constituency development planning in India.

Analyze citizen civic issue submissions and produce a complete structured analysis.
You MUST use the full 1.0-10.0 urgency scale with decimal precision. DO NOT default to 5.0.

=== URGENCY SCORE - CALIBRATION BENCHMARKS ===

Assign a PRECISE DECIMAL score (e.g., 8.5, 9.2, 3.7). Use these anchors:

- 1.0-2.0  MINOR NUISANCES: No real harm. Purely cosmetic or trivial.
           Examples: overgrown grass on footpath, faded road markings, missing park bench.

- 3.0-4.0  LOW-PRIORITY MAINTENANCE: Inconvenient, no safety risk.
           Examples: small pothole on a side street, flickering streetlight in a well-lit area,
           garbage not collected for 2-3 days, minor pavement crack.

- 5.0-6.0  STANDARD MAINTENANCE: Moderate impact. Timely action needed.
           Examples: broken streetlight on a main road, multiple potholes, school washroom out
           of service, blocked roadside drain.

- 7.0-8.0  SERIOUS COMMUNITY IMPACT: Safety/health risk. Days matter.
           Examples: water main broken (colony without supply), sewage overflow on street,
           bridge with visible damage, hospital shortage, large pothole causing accidents.

- 9.0-10.0 CRITICAL EMERGENCY: Immediate danger to life. Hours matter. ALWAYS score 9.0-10.0.
           Examples: dead body reported, serious accident with injuries, building collapse,
           structure about to fall, active fire, gas leak, explosion, people trapped.

HARD RULES - NEVER violate:
  - Death / dead body reported -> score MUST be 10.0, triage = critical_emergency.
  - Collapse / fire / gas leak / explosion -> score MUST be 9.0+, triage = critical_emergency.
  - DO NOT assign 5.0 as a default. Every score must reflect actual severity.
  - Use decimal precision: e.g. 8.5 not 8, 3.7 not 4.

=== TRIAGE CATEGORIES - Assign exactly one ===

Read ALL four definitions carefully before assigning. Getting this wrong is a critical failure.

- quick_fix: Minor, LOCALIZED maintenance items that need no new engineering plans or budget approvals.
  Score typically 2.5-5.5.
  Examples: potholes, trash pile on a street, broken streetlight, clogged roadside drain,
  damaged footpath, missing road sign, broken bench, localized cleaning needed.
  KEY RULE: The problem already EXISTS. It just needs to be REPAIRED or CLEANED.

- urgent_infrastructure: Major utility or structural FAILURES disrupting services for many citizens.
  NOT immediately life-threatening, but causing significant public harm. Score typically 6.5-8.5.
  Examples: burst water main cutting supply to colony, local grid failure (blackout),
  broken sewer line with overflow, severe street flooding blocking access,
  structural damage to an existing bridge (not collapse), hospital supply shortage.
  KEY RULE: An EXISTING asset/service has FAILED and is harming many people.

- long_term_planning: ANY request to CREATE a NEW civic asset or large engineering project.
  Score typically 2.0-5.5.
  KEY RULE: If someone is asking to BUILD or CREATE something that does NOT currently exist,
  it is ALWAYS long_term_planning, regardless of how urgently they phrase it.
  Examples: "need a new college", "build a hospital", "construct a flyover/bridge",
  "establish a new park", "create a community center", "want a new road to be built",
  "we need a university in our area", "our town needs a bus depot".

- critical_emergency: IMMEDIATE, active threats to human life. Score 9.0-10.0.
  Requires emergency services (police, ambulance, fire brigade) right now.
  Examples: dead body found, building collapse in progress, serious accident with casualties,
  active fire, gas leak/explosion, people trapped under debris, drowning victim.

=== LOCATION ===
- Capture specific location in location_hint if mentioned by citizen.
- Only populate geocode if you have strong geographic knowledge. Otherwise null.

Output ONLY the structured JSON. No preamble.
"""


# ─── Gemini Client ────────────────────────────────────────────────────────────

def get_client() -> genai.Client:
    """Local hackathon demo client using only the Gemini Developer API key from .env."""
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _media_part(media: Dict[str, Any]) -> types.Part:
    mime_type = media.get("mime_type") or "application/octet-stream"
    return types.Part.from_bytes(data=media["bytes"], mime_type=mime_type)


# ─── Response Coercers ────────────────────────────────────────────────────────

def _coerce_formalization(response: Any) -> FormalizationResult:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, FormalizationResult):
        return parsed
    if isinstance(parsed, dict):
        return FormalizationResult.model_validate(parsed)
    text = getattr(response, "text", None) or ""
    try:
        return FormalizationResult.model_validate_json(text)
    except (ValidationError, ValueError):
        return FormalizationResult.model_validate(json.loads(text))


def _coerce_deduplication(response: Any) -> DeduplicationResult:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DeduplicationResult):
        return parsed
    if isinstance(parsed, dict):
        return DeduplicationResult.model_validate(parsed)
    text = getattr(response, "text", None) or ""
    try:
        return DeduplicationResult.model_validate_json(text)
    except (ValidationError, ValueError):
        return DeduplicationResult.model_validate(json.loads(text))


def _coerce_analysis(response: Any) -> Dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, SubmissionAnalysis):
        return parsed.model_dump(mode="json")
    if isinstance(parsed, dict):
        return SubmissionAnalysis.model_validate(parsed).model_dump(mode="json")
    text = getattr(response, "text", None) or ""
    try:
        return SubmissionAnalysis.model_validate_json(text).model_dump(mode="json")
    except (ValidationError, ValueError):
        return SubmissionAnalysis.model_validate(json.loads(text)).model_dump(mode="json")


# ─── Step 1: Formalization ────────────────────────────────────────────────────

def formalize_submission(
    raw_text: str,
    media: Optional[List[Dict[str, Any]]] = None,
) -> FormalizationResult:
    """
    Step 1 of the V2.0 intake pipeline.

    Converts raw citizen input into clean formal English by:
    - Translating regional Indian languages (Hindi, Bengali, Tamil, etc.)
    - Removing profanity and offensive language, replacing with neutral factual text
    - Scrubbing slang, transliteration, colloquialisms, and emotional overstatements
    - Detecting the source language

    Never raises — falls back to a safe default on any Gemini failure so the
    intake pipeline never halts and no citizen report is ever lost.
    """
    media = media or []
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = (
        "You are a strict, professional translator and civic editor for an Indian government portal.\n\n"
        "TASK: Convert the raw citizen input below into a clean, formal, government-record-ready English description.\n\n"
        "RAW CITIZEN INPUT:\n"
        "'''\n"
        f"{raw_text or '[No text provided. Analyze the civic issue shown in the attached photo/media and describe it formally.]'}\n"
        "'''\n\n"
        "STRICT MANDATORY RULES — violating any of these is unacceptable:\n"
        "1. TRANSLATE: ALL Indian regional languages (Hindi, Bengali, Tamil, Telugu, Marathi, Punjabi, Gujarati, Malayalam, Kannada, Odia, Urdu, Assamese, Bhojpuri, Rajasthani) "
           "and ALL transliterations (Hinglish, Banglish, Tanglish, Roman-script regional text) MUST be translated into formal English. "
           "The output formal_description MUST be entirely in English.\n"
        "2. SCRUB PROFANITY: Completely remove ALL slang, colloquialisms, abusive terms, and profanities (e.g., 'sala', 'bekar', 'faltu', 'ganda', 'bakwaas', 'ch**ya', 'haramkhor', 'madarchod' and equivalents in any language). "
           "Replace with neutral, objective civic descriptions. The formal_description MUST NOT contain any of the original slang or profanity.\n"
        "3. PRESERVE FACTS: The factual civic complaint — location, problem type, severity, affected people — must be fully preserved and accurately described.\n"
        "4. PROFESSIONAL TONE: Use formal, polite language appropriate for a government record. No emotional language, exaggerations, or first-person anger.\n"
        "5. DETECT LANGUAGE: Report the ISO 639-1 code of the primary language in the original input.\n"
        "6. FLAG PROFANITY: Set profanity_detected=true if ANY profanity, slang, or offensive language was present in the original.\n"
    )

    parts: List[types.Part] = [types.Part.from_text(text=prompt)]
    parts.extend(_media_part(m) for m in media if m.get("bytes"))

    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=FORMALIZATION_SYSTEM,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=FormalizationResult,
            ),
        )
        return _coerce_formalization(response)
    except Exception as e:
        print(f"CRITICAL GEMINI ERROR: {e}")
        # Graceful fallback: preserve raw text, mark as unprocessed
        summary = (raw_text.strip()[:160] if raw_text.strip()
                   else "A civic issue has been reported via submitted media.")
        return FormalizationResult(
            formal_description=raw_text.strip() or "A civic issue has been reported via submitted media.",
            detected_language="en",
            profanity_detected=False,
            cleaned_summary=summary,
        )


# ─── Step 2: Deduplication ────────────────────────────────────────────────────

def check_deduplication(
    formal_description: str,
    address: str,
    existing_issues: List[Dict[str, Any]],
) -> DeduplicationResult:
    """
    Step 2 of the V2.0 intake pipeline.

    Compares the new issue's formal_description and address against up to 20 existing
    active issues using Gemini. Returns a structured duplicate decision.

    Confidence gate: only accepts duplicate decisions with confidence >= 0.85 to
    prevent false merges. Any confidence below this threshold forces a new issue.

    Never raises — always returns DeduplicationResult(is_duplicate=False) on any
    failure so no citizen report is silently dropped.
    """
    if not existing_issues:
        return DeduplicationResult(
            is_duplicate=False,
            master_id=None,
            confidence=1.0,
            reasoning="No existing active issues to compare against — creating a new issue.",
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Build a compact comparison list (only fields Gemini needs, capped at 20)
    comparison_list = [
        {
            "id": str(issue.get("id", "")),
            "description": (
                issue.get("formal_description") or issue.get("text", "")
            )[:250],
            "address": str(issue.get("address") or "")[:120],
            "category": str(issue.get("category", "")),
            "status": str(issue.get("status", "")),
        }
        for issue in existing_issues
        if not issue.get("is_archived")
    ][:20]

    prompt = (
        "Determine whether the following NEW civic issue submission is a duplicate of any existing active issue.\n\n"
        f"NEW SUBMISSION:\n"
        f"  Description: {formal_description[:350]}\n"
        f"  Address: {address or 'Not specified'}\n\n"
        f"EXISTING ACTIVE ISSUES:\n{json.dumps(comparison_list, ensure_ascii=True, indent=2)}\n\n"
        "DEDUPLICATION RULES:\n"
        "• Mark is_duplicate=true ONLY if the new submission describes the EXACT SAME real-world "
        "problem at the SAME physical location as an existing issue.\n"
        "• The same type of problem at a DIFFERENT location is NOT a duplicate.\n"
        "• A broad category match (e.g., both about potholes) at different streets is NOT a duplicate.\n"
        "• If both the problem type AND location strongly overlap, and you are >= 85% confident, "
        "set is_duplicate=true and provide the matching master_id.\n"
        "• When in doubt, set is_duplicate=false — it is safer to create a new issue than to "
        "incorrectly merge two different real-world problems.\n"
    )

    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=DeduplicationResult,
            ),
        )
        result = _coerce_deduplication(response)

        # Enforce minimum confidence threshold — reject ambiguous duplicates
        if result.is_duplicate and result.confidence < 0.85:
            return DeduplicationResult(
                is_duplicate=False,
                master_id=None,
                confidence=result.confidence,
                reasoning=(
                    f"Possible match found but confidence {result.confidence:.2f} is below "
                    "the required 0.85 threshold. Creating a new issue to prevent false merges."
                ),
            )
        return result

    except Exception as exc:
        print(f"CRITICAL GEMINI ERROR: {exc}")
        # Safety fallback: always treat as new issue — never silently drop a report
        return DeduplicationResult(
            is_duplicate=False,
            master_id=None,
            confidence=0.0,
            reasoning=f"Deduplication check failed ({exc!s}). Treating as new issue to prevent any data loss.",
        )


# ─── Step 3: Full Analysis ────────────────────────────────────────────────────

def _local_fallback_analysis(text: str) -> Dict[str, Any]:
    """
    Keyword-based triage fallback when Gemini is unavailable.
    Activated only when ENABLE_LOCAL_AI_FALLBACK=true is set in .env.

    Prints a loud WARNING to the terminal so operators can see the API is failing.
    """
    print(
        "\n" + "=" * 72 + "\n"
        "WARNING: Gemini API failed — using local keyword fallback!\n"
        "Check your GEMINI_API_KEY environment variable and network connectivity.\n"
        "AI scores will be approximate. Fix the API key to restore precision.\n"
        + "=" * 72 + "\n"
    )

    lowered = text.lower()
    category: Category
    triage: TriageCategory

    # ── CRITICAL EMERGENCY detection (MUST run first, highest priority) ──────
    # Any of these keywords indicate an immediate danger to human life.
    emergency_keywords = [
        "dead body", "found dead", "dead man", "dead woman", "dead child",
        "dead", "death", "died", "corpse", "body found",
        "collapse", "collapsed", "collapsing", "building fell", "structure fell",
        "accident", "road accident", "car crash", "vehicle crash", "bike crash",
        "fire", "burning", "on fire",
        "gas leak", "gas cylinder", "lpg leak",
        "explosion", "blast",
        "injured", "serious injury", "critical condition",
        "drowning", "drowned",
        "murder", "killed", "stabbed", "shot",
        "trapped", "rescue needed", "missing person",
        "electrocuted", "electric shock",
    ]
    if any(w in lowered for w in emergency_keywords):
        summary = text.strip()[:180] or "Critical emergency reported."
        print(
            f"WARNING [fallback]: Emergency keyword detected in submission.\n"
            f"  Text preview: \"{summary[:100]}\"\n"
            f"  Assigning: triage=critical_emergency, urgency=10/10\n"
        )
        return SubmissionAnalysis(
            summary=summary,
            category="safety",
            triage_category="critical_emergency",
            urgency_score=10.0,
            sentiment="negative",
            suggested_department="Emergency Services / Police / District Magistrate",
            constituency_priority=(
                "CRITICAL EMERGENCY: Immediate life-threatening situation reported. "
                "Requires emergency services and district-level escalation NOW."
            ),
            keywords=["emergency", "critical", "life-threatening", "immediate-action"],
            location_hint=None,
            geocode=None,
            rationale=(
                "Emergency keywords detected by local fallback analysis. "
                "Escalated to maximum urgency. AI API was unavailable for full analysis."
            ),
        ).model_dump(mode="json")

    # ── Tier 1: long_term_planning — NEW asset creation keywords ────────
    long_term_words = [
        "college", "university", "build", "construct", "flyover",
        "new bridge", "new hospital", "new school", "new road", "new park",
        "establish", "create a", "need a new", "build a", "new clinic",
        "new community", "new market", "bus depot", "metro", "new highway",
        "new drainage", "new pipeline", "expand", "develop a",
    ]
    # ── Tier 2: urgent_infrastructure — existing service failures ────────
    urgent_words = [
        "burst pipe", "burst water", "flooding", "flood", "sewer",
        "sewage overflow", "power grid", "blackout", "grid failure",
        "power failure", "no water supply", "water contamination",
        "structural damage", "damaged bridge", "hospital shortage",
        "broken sewer", "no electricity", "transformer failure",
    ]

    # ── Standard keyword triage (Tier 1 > Tier 2 > Tier 3) ──────────────
    if any(w in lowered for w in long_term_words):
        # Determine best category for long-term request
        if any(w in lowered for w in ["college", "university", "school", "classroom"]):
            category = "education"
        elif any(w in lowered for w in ["hospital", "clinic", "health"]):
            category = "health"
        elif any(w in lowered for w in ["bridge", "flyover", "road", "highway"]):
            category = "roads"
        elif any(w in lowered for w in ["park", "garden", "playground"]):
            category = "environment"
        else:
            category = "other"
        triage = "long_term_planning"
        urgency = 3.2
    elif any(w in lowered for w in urgent_words):
        if any(w in lowered for w in ["water", "pipeline", "tap"]):
            category = "water"
        elif any(w in lowered for w in ["sewer", "sewage", "drainage"]):
            category = "sanitation"
        elif any(w in lowered for w in ["power", "electricity", "blackout", "grid"]):
            category = "electricity"
        elif any(w in lowered for w in ["bridge", "structural"]):
            category = "roads"
        elif any(w in lowered for w in ["hospital", "health"]):
            category = "health"
        else:
            category = "other"
        triage = "urgent_infrastructure"
        urgency = 7.8
    elif any(w in lowered for w in ["sewage", "sewer"]):
        category, triage, urgency = "sanitation", "urgent_infrastructure", 7.2
    elif any(w in lowered for w in ["garbage", "waste", "litter", "trash", "dump"]):
        category, triage, urgency = "sanitation", "quick_fix", 3.6
    elif any(w in lowered for w in ["pothole", "road surface", "broken road", "damaged road"]):
        category, triage, urgency = "roads", "quick_fix", 4.2
    elif any(w in lowered for w in ["road", "traffic", "street", "footpath", "pavement"]):
        category, triage, urgency = "roads", "quick_fix", 3.8
    elif any(w in lowered for w in ["streetlight", "street light", "lamp post"]):
        category, triage, urgency = "electricity", "quick_fix", 3.4
    elif any(w in lowered for w in ["water", "pipeline", "tap", "leakage"]):
        category, triage, urgency = "water", "quick_fix", 4.5
    elif any(w in lowered for w in ["flood", "waterlogging"]):
        category, triage, urgency = "water", "urgent_infrastructure", 7.4
    elif any(w in lowered for w in ["school", "teacher", "classroom", "education"]):
        category, triage, urgency = "education", "quick_fix", 4.8
    elif any(w in lowered for w in ["park", "garden", "playground"]):
        category, triage, urgency = "environment", "long_term_planning", 2.8
    else:
        category, triage, urgency = "other", "quick_fix", 3.9

    summary = text.strip()[:180] or "Citizen submitted a media-only civic issue."
    return SubmissionAnalysis(
        summary=summary,
        category=category,
        triage_category=triage,
        urgency_score=urgency,
        sentiment="negative" if urgency >= 7 else "neutral",
        suggested_department="Constituency development office",
        constituency_priority="Requires triage against demographic vulnerability and citizen demand.",
        keywords=[category, triage, "citizen-report", "fallback-analysis"],
        location_hint=None,
        geocode=None,
        rationale="Generated by local keyword fallback because Gemini was unavailable. Accuracy may be lower.",
    ).model_dump(mode="json")


def analyze_submission(
    text: str = "",
    media: Optional[List[Dict[str, Any]]] = None,
    demographics_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Step 3 of the V2.0 intake pipeline (unique new issues only).

    Performs full AI analysis: category, triage_category, urgency_score, department,
    constituency_priority, keywords, geocode, and all other structured fields.

    `text` should be the already-formalized formal_description from Step 1.
    Media entries: {"bytes": b"...", "mime_type": "image/jpeg", "path": "static/uploads/f.jpg"}
    """
    media = media or []
    demographics_context = demographics_context or []
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = (
        "Analyze this civic issue submission and return a complete structured analysis.\n\n"
        f"ISSUE DESCRIPTION (formal English):\n{text or '[No text — analyze the attached photo/media.]'}\n\n"
        f"CONSTITUENCY DEMOGRAPHIC CONTEXT:\n{json.dumps(demographics_context[:5], ensure_ascii=True)}\n\n"
        "Assign the correct category, triage_category, urgency_score, and all other required fields."
    )

    parts: List[types.Part] = [types.Part.from_text(text=prompt)]
    parts.extend(_media_part(m) for m in media if m.get("bytes"))

    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=ANALYSIS_SYSTEM,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=SubmissionAnalysis,
            ),
        )
        return _coerce_analysis(response)
    except Exception as e:
        print(f"CRITICAL GEMINI ERROR: {e}")
        if os.getenv("ENABLE_LOCAL_AI_FALLBACK", "false").lower() == "true":
            return _local_fallback_analysis(text)
        raise


# ─── Demographics ─────────────────────────────────────────────────────────────

DEFAULT_DEMOGRAPHICS: List[Dict[str, Any]] = [
    {
        "ward_id": "unknown",
        "ward_name": "Unknown Ward",
        "population": 0,
        "households": 0,
        "vulnerability_index": 0,
        "water_access_pct": None,
        "sanitation_access_pct": None,
        "health_facility_count": None,
    }
]


def _load_demographics_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return DEFAULT_DEMOGRAPHICS
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_demographics() -> Dict[str, Any]:
    """
    Load constituency demographics strictly from the local CSV.
    Never attempts BigQuery or any network connection.
    """
    return {
        "source": "csv_local",
        "path": str(LOCAL_DEMOGRAPHICS_CSV),
        "rows": _load_demographics_csv(LOCAL_DEMOGRAPHICS_CSV),
    }


def fetch_demographics_from_bigquery(
    constituency_id: Optional[str] = None,
    fallback_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for the legacy function name.
    All demographic data comes strictly from the local CSV — no cloud calls.
    """
    del constituency_id, fallback_csv_path
    return load_demographics()
