"""
Healthcare chatbot with layered guardrails.

Guardrails:
1. Crisis routing  — self-harm language goes to support resources, not a refusal
2. Topic blocking  — genuinely off-topic / abusive requests are stopped
3. PII redaction   — emails and card numbers scrubbed from input
4. Human approval  — appointment booking requires confirmation
5. Output disclaimer — health information carries a consult-your-provider note
"""

from __future__ import annotations

import re
from typing import Any
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------------
# Guardrail 1 + 2: crisis routing and topic blocking
# --------------------------------------------------------------------------


CRISIS_TERMS = [
    "kill myself",
    "killing myself",
    "end my life",
    "want to die",
    "take my own life",
    "hurt myself",
    "harm myself",
    "cut myself",
    "cutting myself",
    "suicidal",
]

BLOCKED_TOPICS = [
    # illicit manufacture
    "how to make meth",
    "cook meth",
    "synthesize fentanyl",
    "make a bomb",
    "build a bomb",
    "explosive device",
    # harm to others
    "poison someone",
    "kill someone",
    "harm someone",
    "untraceable poison",
    # misuse of the system
    "bypass security",
    "sql injection",
    "exploit the system",
    # prompt injection
    "ignore previous instructions",
    "ignore your instructions",
    "disregard your system prompt",
    "pretend you are",
]

CRISIS_RESPONSE = (
    "I'm really glad you reached out. I'm not able to help with this myself, but "
    "people who can are available right now — you can call or text 988 to reach the "
    "Suicide & Crisis Lifeline, any time of day.\n\n"
    "If you're in immediate danger, please call 911.\n\n"
    "You don't have to go through this alone."
)

OFF_TOPIC_RESPONSE = (
    "I'm a healthcare assistant, so I can only help with medical questions, "
    "medication information, and appointments. Is there something health-related "
    "I can help you with?"
)


def _compile(terms: list[str]) -> list[re.Pattern]:
    """Word-boundary patterns so 'hack' never matches 'hacking cough'."""
    return [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in terms]


class HealthCareSafetyFilter(AgentMiddleware):
    """Route crisis language to support; block off-topic or abusive requests."""

    _crisis = _compile(CRISIS_TERMS)
    _blocked = _compile(BLOCKED_TOPICS)

    @staticmethod
    def _latest_human_text(state: AgentState) -> str | None:
        """Most recent human turn, or None. Checking messages[0] would only ever
        inspect the first turn of a persisted thread."""
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) == "human":
                content = message.content
                # Content can be a list of blocks for multimodal input.
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                    )
                return str(content)
        return None

    @staticmethod
    def _halt(text: str) -> dict[str, Any]:
        return {
            "messages": [{"role": "assistant", "content": text}],
            "jump_to": "end",
        }

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        text = self._latest_human_text(state)
        if not text:
            return None

        # Crisis check runs first and returns support, not a refusal.
        if any(pattern.search(text) for pattern in self._crisis):
            return self._halt(CRISIS_RESPONSE)

        if any(pattern.search(text) for pattern in self._blocked):
            return self._halt(OFF_TOPIC_RESPONSE)

        return None


# --------------------------------------------------------------------------
# Guardrail 5: output disclaimer
# --------------------------------------------------------------------------


class MedicalOutputValidator(AgentMiddleware):
    """Append a consult-your-provider note to substantive health responses."""

    DISCLAIMER = (
        "\n\n*This is general health information, not medical advice. "
        "Please consult a qualified healthcare professional for guidance specific "
        "to your situation.*"
    )

    # A phrase from DISCLAIMER itself, not one the model says naturally.
    MARKER = "not medical advice"

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        # Tool-call turns carry no prose for the patient to read.
        if getattr(last, "tool_calls", None):
            return None

        content = last.content
        if not isinstance(content, str) or not content.strip():
            return None

        # Crisis and off-topic replies are complete as written.
        if content.strip() in (CRISIS_RESPONSE, OFF_TOPIC_RESPONSE):
            return None

        if self.MARKER in content.lower():
            return None

        # Return a replacement rather than mutating in place, so the change is
        # recorded in state and survives checkpointing.
        return {
            "messages": [
                AIMessage(content=content + self.DISCLAIMER, id=last.id)
            ]
        }


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@tool
def search_symptoms(symptoms: str) -> str:
    """Search for general information about medical symptoms."""
    return (
        f"General information about: {symptoms}. "
        "A clinician is needed for diagnosis."
    )


@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment for a patient."""
    return f"Appointment booked for {patient_name} with Dr. {doctor} on {date}."


@tool
def get_medication_info(medication: str) -> str:
    """Get general information about a medication."""
    return (
        f"General information about {medication}. "
        "Consult your pharmacist or prescriber about dosing and interactions."
    )


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a healthcare information assistant for a medical practice. \
You help patients find general health information, look up medications, and schedule appointments.

## Scope

You CAN:
- Explain symptoms, conditions, and medical terminology in plain language
- Provide general information about medications: what they treat, common side effects, general precautions
- Help patients book, confirm, or ask about appointments
- Help patients understand what kind of care they may need (urgent care vs. primary care vs. specialist)

You CANNOT and MUST NOT:
- Diagnose. Never tell a patient what condition they have, even when the symptoms seem obvious. \
Describe possibilities and direct them to a clinician.
- Recommend, adjust, or calculate medication doses, or advise starting or stopping any medication. \
Direct dosing questions to their prescriber or pharmacist.
- Interpret lab results, imaging, or test values as applied to a specific patient.
- Advise on whether to seek treatment, delay treatment, or decline a clinician's recommendation.

If a request falls outside healthcare entirely, say so briefly and redirect. Do not answer \
off-topic questions even when you know the answer.

## Emergency escalation — highest priority

If a patient describes symptoms that could indicate a medical emergency — including chest pain \
or pressure, difficulty breathing, sudden weakness or numbness (especially one-sided), trouble \
speaking, sudden severe headache, uncontrolled bleeding, loss of consciousness, or a severe \
allergic reaction — your FIRST and PRIMARY response is to direct them to call 911 or go to the \
nearest emergency department immediately.

Do not gather more details first. Do not call tools first. Do not soften this with caveats. \
Lead with the instruction to seek emergency care, then briefly explain why.

## Mental health

If a patient expresses distress, respond with warmth and without judgment. Do not attempt to \
assess risk, and do not ask screening questions. Direct them to the 988 Suicide & Crisis \
Lifeline (call or text 988), or 911 if they are in immediate danger.

## Using tools

- Use `search_symptoms` when a patient describes symptoms and wants information.
- Use `get_medication_info` for questions about a specific medication.
- Use `book_appointment` only when the patient has clearly asked to schedule, and only after \
you have their name, a preferred date, and a doctor. Ask for anything missing before calling \
the tool — never guess or fill in placeholder values.
- Never invent details you don't have. If a tool returns nothing useful, say so.

## Handling uncertainty

If you do not know something, say so. Never fabricate medical facts, drug interactions, clinic \
policies, provider names, or availability. Speculation in a healthcare context causes real harm, \
and "I'm not sure — your pharmacist can confirm this" is always an acceptable answer.

If a patient's question is ambiguous in a way that matters clinically, ask one clarifying \
question rather than assuming.

## Privacy

Do not ask for more personal information than a task requires. Never ask for Social Security \
numbers, insurance ID numbers, or payment card details. If a patient volunteers sensitive \
information, do not repeat it back.

## Tone

Write in plain language at roughly an eighth-grade reading level. Define medical terms when you \
use them. Be warm and concrete — patients asking these questions are often worried. Keep \
responses short; two or three short paragraphs is usually enough. Do not use alarming language, \
and do not minimize concerns either."""


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

healthcare_bot = create_agent(
    model="gpt-5-mini",
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # 1 + 2: crisis routing and topic blocking (runs before the model)
        HealthCareSafetyFilter(),
        # 3: scrub PII from patient input
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        # 4: human approval before booking
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_appointment": True,
                "search_symptoms": False,
                "get_medication_info": False,  # was get_medication_information
            }
        ),
        # 5: disclaimer on outbound health information
        MedicalOutputValidator(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=SYSTEM_PROMPT,
)