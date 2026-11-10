
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "gemini_config.json"
)

DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "gemini_vlm_contract.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_api_key(secret_name: str) -> str:

    # Streamlit Cloud secret.
    try:
        import streamlit as st
        value = st.secrets.get(secret_name)
    except Exception:
        value = None

    # Colab/local environment fallback.
    value = value or os.environ.get(secret_name)

    if not value:
        raise RuntimeError(
            f"{secret_name} is missing. "
            "Configure it in Streamlit Cloud Secrets."
        )

    return str(value).strip()


def _build_prompt(contract: dict[str, Any]) -> str:

    schema = contract["output_schema"]

    return f"""
You are an infrastructure visual evidence extraction system.

Analyze the supplied infrastructure image only for observable
visual evidence relevant to infrastructure condition assessment.

Return ONLY valid JSON.

Do not provide a final engineering diagnosis.
Do not claim structural safety.
Do not invent measurements.
Do not infer hidden damage that is not visually supported.
If evidence is insufficient, explicitly report insufficient evidence.

Required JSON structure:

{json.dumps(schema, indent=2)}

The engineering_decision field MUST remain:

{{
  "status": "not_provided",
  "reason": "Engineering significance and final disposition require qualified human validation."
}}
"""


def _validate_result(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:

    required_fields = [
        "image_assessment",
        "observations",
        "negative_evidence",
        "limitations",
        "engineering_decision",
    ]

    missing = [
        field for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            f"Gemini response missing fields: {missing}"
        )

    if not isinstance(result["image_assessment"], dict):
        raise ValueError("image_assessment must be an object.")

    if not isinstance(result["observations"], list):
        raise ValueError("observations must be a list.")

    if not isinstance(result["negative_evidence"], list):
        raise ValueError("negative_evidence must be a list.")

    if not isinstance(result["limitations"], list):
        raise ValueError("limitations must be a list.")

    decision = result["engineering_decision"]

    if not isinstance(decision, dict):
        raise ValueError("engineering_decision must be an object.")

    if decision.get("status") != "not_provided":
        raise ValueError(
            "Gemini attempted to provide an engineering decision."
        )

    observation_fields = [
        "observation",
        "evidence_type",
        "location_description",
        "visual_confidence",
    ]

    for index, observation in enumerate(result["observations"]):

        if not isinstance(observation, dict):
            raise ValueError(
                f"Observation {index} must be an object."
            )

        missing = [
            field for field in observation_fields
            if field not in observation
        ]

        if missing:
            raise ValueError(
                f"Observation {index} missing fields: {missing}"
            )

    return {
        "source": "gemini_vlm",
        "model": None,
        "image_assessment": result["image_assessment"],
        "observations": result["observations"],
        "negative_evidence": result["negative_evidence"],
        "limitations": result["limitations"],
        "engineering_decision": {
            "status": "not_provided",
            "reason": (
                "Engineering significance and final disposition "
                "require qualified human validation."
            ),
        },
    }


def _status_code(exc: Exception) -> int | None:

    candidates = [
        getattr(exc, "code", None),
        getattr(exc, "status_code", None),
    ]

    response = getattr(exc, "response", None)

    if response is not None:
        candidates.append(
            getattr(response, "status_code", None)
        )

    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue

    return None


def _is_model_unavailable(exc: Exception) -> bool:

    status = _status_code(exc)
    message = str(exc).lower()

    if status == 404:
        return True

    return status == 400 and any(
        phrase in message
        for phrase in (
            "model not found",
            "model is not found",
            "unsupported model",
            "unknown model",
            "not supported for generatecontent",
        )
    )


def _is_auth_error(exc: Exception) -> bool:

    status = _status_code(exc)
    message = str(exc).lower()

    if status in (401, 403):
        return True

    return status == 400 and any(
        phrase in message
        for phrase in (
            "invalid api key",
            "api key not valid",
            "api_key_invalid",
            "permission denied",
            "not authorized",
        )
    )


def _is_transient_error(
    exc: Exception,
    retry_codes: set[int],
) -> bool:

    status = _status_code(exc)

    if status in retry_codes:
        return True

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    name = type(exc).__name__.lower()

    return any(
        term in name
        for term in (
            "timeout",
            "connecterror",
            "connectionerror",
            "networkerror",
            "remoteprotocolerror",
        )
    )


def _mime_type(image_bytes: bytes) -> str:

    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"

    return "image/jpeg"


def _parse_json_response(response_text: str) -> dict[str, Any]:

    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)

    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            raise

        parsed = json.loads(cleaned[start:end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Gemini JSON response must be an object.")

    return parsed


def _error_label(exc: Exception) -> str:

    status = _status_code(exc)

    if status is not None:
        return f"{type(exc).__name__} (HTTP {status})"

    return type(exc).__name__


def analyze_image_with_gemini(
    image_bytes: bytes,
    config_path: Path | None = None,
    contract_path: Path | None = None,
) -> dict[str, Any]:

    if not image_bytes:
        raise ValueError("image_bytes cannot be empty.")

    config_path = config_path or DEFAULT_CONFIG_PATH
    contract_path = contract_path or DEFAULT_CONTRACT_PATH

    config = _load_json(config_path)
    contract = _load_json(contract_path)

    key_name = config.get(
        "api_key_environment_variable",
        "GEMINI_API_KEY",
    )

    api_key = _get_api_key(key_name)

    primary_model = config["model"]

    fallback_models = config.get(
        "fallback_models",
        [],
    )

    model_chain = list(
        dict.fromkeys([primary_model, *fallback_models])
    )

    policy = config.get("retry_policy", {})

    retry_enabled = bool(
        policy.get("enabled", True)
    )

    max_attempts = int(
        policy.get("max_attempts_per_model", 2)
    )

    if not retry_enabled:
        max_attempts = 1

    max_attempts = max(1, min(max_attempts, 4))

    max_total_attempts = max(
        1,
        int(policy.get("max_total_attempts", 8)),
    )

    base_delay = float(
        policy.get("base_delay_seconds", 1.2)
    )

    max_delay = float(
        policy.get("max_delay_seconds", 5.0)
    )

    jitter = float(
        policy.get("jitter_seconds", 0.5)
    )

    retry_codes = set(
        int(code)
        for code in policy.get(
            "retry_status_codes",
            [408, 425, 429, 500, 502, 503, 504],
        )
    )

    timeout_seconds = int(
        config.get("timeout_seconds", 45)
    )

    # Disable SDK-level retries so our retry budget is predictable.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(
                attempts=1
            ),
        ),
    )

    prompt = _build_prompt(contract)

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=_mime_type(image_bytes),
    )

    failures = []
    total_attempts = 0

    try:

        for model_name in model_chain:

            for attempt in range(1, max_attempts + 1):

                if total_attempts >= max_total_attempts:
                    break

                total_attempts += 1

                try:

                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            image_part,
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            temperature=float(
                                config.get("temperature", 0.1)
                            ),
                            max_output_tokens=int(
                                config.get("max_output_tokens", 1024)
                            ),
                            response_mime_type="application/json",
                        ),
                    )

                    response_text = getattr(
                        response,
                        "text",
                        None,
                    )

                    if not response_text:
                        raise RuntimeError(
                            "Gemini returned an empty response."
                        )

                    parsed = _parse_json_response(
                        response_text
                    )

                    validated = _validate_result(
                        parsed,
                        contract,
                    )

                    validated["model"] = model_name

                    return validated

                except Exception as exc:

                    label = _error_label(exc)

                    # Invalid credentials cannot be repaired by
                    # changing the model.
                    if _is_auth_error(exc):
                        raise RuntimeError(
                            "Gemini authentication or permission failed. "
                            "Check the API key and Gemini API access."
                        ) from exc

                    # Model unavailable: move to the next model.
                    if _is_model_unavailable(exc):
                        failures.append(
                            f"{model_name}: {label}"
                        )
                        break

                    # Retry transient service/network errors.
                    if _is_transient_error(exc, retry_codes):

                        if attempt < max_attempts:

                            delay = min(
                                max_delay,
                                base_delay * (2 ** (attempt - 1)),
                            )

                            time.sleep(
                                delay + random.uniform(
                                    0.0,
                                    max(0.0, jitter),
                                )
                            )

                            continue

                        failures.append(
                            f"{model_name}: {label}"
                        )
                        break

                    # A non-model-related HTTP 400 is a request
                    # problem, not a reason to retry another model.
                    if _status_code(exc) == 400:
                        raise RuntimeError(
                            "Gemini rejected the request (HTTP 400). "
                            "Check image format and request configuration."
                        ) from exc

                    # Invalid JSON/schema or other model output issue:
                    # move to the next model.
                    failures.append(
                        f"{model_name}: {label}"
                    )
                    break

            if total_attempts >= max_total_attempts:
                break

        tried = ", ".join(model_chain)

        last_failure = (
            failures[-1]
            if failures
            else "No model returned usable output."
        )

        raise RuntimeError(
            "Gemini unavailable after bounded retry/fallback. "
            f"Models tried: {tried}. "
            f"Last failure: {last_failure}. "
            "CV evidence should remain available."
        )

    finally:

        try:
            client.close()
        except Exception:
            pass
