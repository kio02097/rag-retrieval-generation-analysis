"""Load text/YAML prompts and split their system/user roles."""
import os
import yaml
from pathlib import Path

def load_prompt_text_from_yaml(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

    raw = Path(path).read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(raw)
    except Exception:
        return raw

    if isinstance(data, dict):
        for key in ["template", "prompt", "text", "content"]:
            if key in data and isinstance(data[key], str) and data[key].strip():
                return data[key]

        if "messages" in data and isinstance(data["messages"], list):
            parts = []
            for m in data["messages"]:
                if isinstance(m, dict):
                    c = m.get("content") or m.get("template") or ""
                    if isinstance(c, str) and c.strip():
                        parts.append(c.strip())
            if parts:
                return "\n\n".join(parts)

    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                c = item.get("content") or item.get("template") or ""
                if isinstance(c, str) and c.strip():
                    parts.append(c.strip())
        if parts:
            return "\n\n".join(parts)

    return raw

def split_system_user(prompt_text: str):
    """
    prompt_text 안에 [SYSTEM] ... [User]/[USER] ... 가 있으면 분리.
    없으면 system="" / user=prompt_text 로 둔다.
    """
    t = prompt_text or ""
    if "[SYSTEM]" in t and "[User]" in t:
        sys_part = t.split("[SYSTEM]", 1)[1].split("[User]", 1)[0].strip()
        user_part = t.split("[User]", 1)[1].strip()
        return sys_part, user_part
    if "[SYSTEM]" in t and "[USER]" in t:
        sys_part = t.split("[SYSTEM]", 1)[1].split("[USER]", 1)[0].strip()
        user_part = t.split("[USER]", 1)[1].strip()
        return sys_part, user_part
    return "", t.strip()
