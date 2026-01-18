import os
import json
from typing import TypedDict, Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = ChatOpenAI(
    openai_api_key=os.environ["DeepSeek_API"],
    openai_api_base="https://api.deepseek.com",
    model_name="deepseek-chat",
)


class TranslationState(TypedDict, total=False):
    query: str
    source_text: str
    target_lang: Literal["en", "ja", "fr"]
    translation: str
    evaluation_ok: bool
    evaluation_feedback: str
    attempts: int
    max_attempts: int


def _detect_target_lang_rule(query: str) -> Optional[str]:
    q = query.lower()
    if "法语" in query or "法文" in query or "french" in q or "français" in q:
        return "fr"
    if "日语" in query or "日文" in query or "japanese" in q or "にほんご" in q:
        return "ja"
    if "英语" in query or "英文" in query or "english" in q:
        return "en"
    return None


def intent_node(state: TranslationState) -> TranslationState:
    query = state["query"]
    rule = _detect_target_lang_rule(query)
    if rule:
        target_lang = rule
    else:
        resp = model.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a classifier. Output exactly one token: en, ja, or fr.",
                },
                {
                    "role": "user",
                    "content": f"Decide target language for translation request:\n{query}",
                },
            ]
        ).content.strip().lower()
        if resp not in {"en", "ja", "fr"}:
            target_lang = "en"
        else:
            target_lang = resp

    resp2 = model.invoke(
        [
            {"role": "system", "content": "Extract only the text to be translated."},
            {"role": "user", "content": query},
        ]
    ).content.strip()

    return {
        "target_lang": target_lang,
        "source_text": resp2,
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
        "evaluation_ok": False,
        "evaluation_feedback": "",
    }


def _translate(state: TranslationState, target_lang: str) -> TranslationState:
    source_text = state["source_text"]
    feedback = state.get("evaluation_feedback", "").strip()
    attempts = int(state.get("attempts", 0)) + 1
    if feedback:
        user_content = (
            f"Text:\n{source_text}\n\nTarget language: {target_lang}\n\n"
            f"Revise using this feedback:\n{feedback}\n\nOutput only the translation."
        )
    else:
        user_content = (
            f"Text:\n{source_text}\n\nTarget language: {target_lang}\n\nOutput only the translation."
        )
    translation = model.invoke(
        [{"role": "system", "content": "You are a professional translator."}, {"role": "user", "content": user_content}]
    ).content.strip()
    return {"translation": translation, "attempts": attempts}


def translate_en_node(state: TranslationState) -> TranslationState:
    return _translate(state, "English")


def translate_ja_node(state: TranslationState) -> TranslationState:
    return _translate(state, "Japanese")


def translate_fr_node(state: TranslationState) -> TranslationState:
    return _translate(state, "French")


def eval_node(state: TranslationState) -> TranslationState:
    source_text = state["source_text"]
    target_lang = state["target_lang"]
    translation = state["translation"]
    resp = model.invoke(
        [
            {"role": "system", "content": "You are a strict translation evaluator. Respond with valid JSON only."},
            {
                "role": "user",
                "content": (
                    f"Target language: {target_lang}\n\n"
                    f"Source:\n{source_text}\n\n"
                    f"Translation:\n{translation}\n\n"
                    'Return JSON: {"ok": boolean, "feedback": string}. '
                    "If ok is false, feedback must be actionable."
                ),
            },
        ]
    ).content.strip()
    try:
        data = json.loads(resp)
        ok = bool(data.get("ok", False))
        feedback = str(data.get("feedback", "")).strip()
    except Exception:
        ok = False
        feedback = f"Evaluator output was not valid JSON: {resp}"
    return {"evaluation_ok": ok, "evaluation_feedback": feedback}


def route_after_intent(state: TranslationState) -> str:
    return state["target_lang"]


def route_after_eval(state: TranslationState) -> str:
    if state.get("evaluation_ok"):
        return "end"
    if int(state.get("attempts", 0)) >= int(state.get("max_attempts", 3)):
        return "end"
    return state["target_lang"]


workflow = StateGraph(TranslationState)
workflow.add_node("intent", intent_node)
workflow.add_node("translate_en", translate_en_node)
workflow.add_node("translate_ja", translate_ja_node)
workflow.add_node("translate_fr", translate_fr_node)
workflow.add_node("eval", eval_node)

workflow.add_edge(START, "intent")
workflow.add_conditional_edges("intent", route_after_intent, {"en": "translate_en", "ja": "translate_ja", "fr": "translate_fr"})

workflow.add_edge("translate_en", "eval")
workflow.add_edge("translate_ja", "eval")
workflow.add_edge("translate_fr", "eval")

workflow.add_conditional_edges(
    "eval",
    route_after_eval,
    {"en": "translate_en", "ja": "translate_ja", "fr": "translate_fr", "end": END},
)

app = workflow.compile()

def main() -> None:
    query = input("query: ").strip()
    result = app.invoke({"query": query, "attempts": 0, "max_attempts": 3})
    print(result.get("translation", ""))
    if not result.get("evaluation_ok", False):
        print(result.get("evaluation_feedback", ""))


if __name__ == "__main__":
    main()
