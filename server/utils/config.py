import base64
import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI


def _required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} 환경변수가 필요합니다.")
    return value


def get_llm(temperature: float = 0.1):
    """Shared Azure OpenAI LangChain client for all text agents."""
    return AzureChatOpenAI(
        azure_endpoint=_required("AOAI_ENDPOINT"),
        api_key=_required("AOAI_API_KEY"),
        azure_deployment=(os.getenv("AOAI_DEPLOY_GPT4O") or os.getenv("AOAI_DEPLOY_GPT4O_MINI") or "gpt-4o-mini"),
        api_version=os.getenv("AOAI_API_VERSION", "2024-10-21"),
        temperature=temperature,
    )


def generate_text(prompt: str) -> str:
    """All service LLM calls go through LangChain."""
    response = get_llm().invoke([
        SystemMessage(content="You are a reliable educational AI agent. Follow the requested output format exactly."),
        HumanMessage(content=prompt),
    ])
    content = response.content
    if isinstance(content, list):
        return "".join(
            str(x.get("text", "")) if isinstance(x, dict) else str(x)
            for x in content
        )
    return str(content or "")


def generate_json(prompt: str) -> dict:
    text = generate_text(prompt + "\nReturn valid JSON only. No markdown fences.").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


def analyze_image(image_bytes: bytes, mime_type: str, prompt: str, system_instruction: str = "") -> str:
    """Analyze uploaded vocabulary material with Azure OpenAI GPT-4o vision."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"

    llm = get_llm(temperature=0.2)
    messages = []
    if system_instruction:
        messages.append(SystemMessage(content=system_instruction))
    messages.append(HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
    ]))
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        return "".join(
            str(x.get("text", "")) if isinstance(x, dict) else str(x)
            for x in content
        )
    return str(content or "")
