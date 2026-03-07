"""CapSolver-based CAPTCHA solving tool for browser-use agents.

Exposes a single `solve_captcha` action via the browser_use Tools API.
The agent detects the CAPTCHA type and sitekey using JavaScript, then calls
this tool to obtain a token from CapSolver, then injects the token via JS.
All blocking network I/O runs synchronously; browser_use runs tool actions
in a thread so this does not block the asyncio event loop.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

from browser_use import Tools

from applypilot import config

logger = logging.getLogger(__name__)

CAPSOLVER_BASE = "https://api.capsolver.com"

# Map the agent-visible type names to CapSolver task types.
TASK_TYPE_MAP: dict[str, str] = {
    "hcaptcha": "HCaptchaTaskProxyLess",
    "recaptchav2": "ReCaptchaV2TaskProxyLess",
    "recaptchav3": "ReCaptchaV3TaskProxyLess",
    "turnstile": "AntiTurnstileTaskProxyLess",
    "funcaptcha": "FunCaptchaTaskProxyLess",
}

# Poll up to 15 times (3 s each = 45 s max wait).
MAX_POLLS = 15


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def build_captcha_tools() -> Tools:
    """Return a Tools instance with the solve_captcha action registered.

    Reads CAPSOLVER_API_KEY from the environment at call time so that the
    caller can load .env before calling this function.
    """
    config.load_env()
    capsolver_key = os.environ.get("CAPSOLVER_API_KEY", "")

    tools = Tools()

    @tools.action(
        description=(
            "Solve a CAPTCHA using the CapSolver API. "
            "Call this after detecting a CAPTCHA on the page — do NOT try to call "
            "CapSolver yourself via JavaScript fetch(). "
            "captcha_type must be exactly one of: hcaptcha, recaptchav2, recaptchav3, "
            "turnstile, funcaptcha. "
            "page_url is the current page URL. site_key is the sitekey from the CAPTCHA element. "
            "page_action is optional (recaptchav3 only, defaults to 'submit'). "
            "turnstile_action and turnstile_cdata are optional (Turnstile only). "
            "Returns the solved token string on success, or a message starting with "
            "'ERROR:' if solving failed — fall back to manual CAPTCHA solving in that case."
        )
    )
    def solve_captcha(
        captcha_type: str,
        page_url: str,
        site_key: str,
        page_action: str = "submit",
        turnstile_action: str = "",
        turnstile_cdata: str = "",
    ) -> str:
        if not capsolver_key:
            return "ERROR: CAPSOLVER_API_KEY not configured — use manual fallback"

        task_type = TASK_TYPE_MAP.get(captcha_type.lower())
        if not task_type:
            valid = ", ".join(TASK_TYPE_MAP)
            return f"ERROR: Unknown captcha_type '{captcha_type}'. Must be one of: {valid}"

        task: dict = {
            "type": task_type,
            "websiteURL": page_url,
            "websiteKey": site_key,
        }

        if captcha_type.lower() == "recaptchav3":
            task["pageAction"] = page_action or "submit"

        if captcha_type.lower() == "turnstile":
            metadata: dict = {}
            if turnstile_action:
                metadata["action"] = turnstile_action
            if turnstile_cdata:
                metadata["cdata"] = turnstile_cdata
            if metadata:
                task["metadata"] = metadata

        # --- Step 1: create task ---
        try:
            create_resp = _post_json(
                f"{CAPSOLVER_BASE}/createTask",
                {"clientKey": capsolver_key, "task": task},
            )
        except Exception as exc:
            return f"ERROR: createTask request failed: {exc}"

        if create_resp.get("errorId", 1) != 0:
            desc = create_resp.get("errorDescription", "")
            return f"ERROR: CapSolver createTask errorId={create_resp['errorId']}: {desc}"

        task_id = create_resp.get("taskId")
        if not task_id:
            return f"ERROR: No taskId in CapSolver response: {create_resp}"

        # --- Step 2: poll for result ---
        for attempt in range(1, MAX_POLLS + 1):
            time.sleep(3)
            try:
                poll_resp = _post_json(
                    f"{CAPSOLVER_BASE}/getTaskResult",
                    {"clientKey": capsolver_key, "taskId": task_id},
                )
            except Exception as exc:
                return f"ERROR: getTaskResult request failed (attempt {attempt}): {exc}"

            if poll_resp.get("errorId", 0) != 0:
                desc = poll_resp.get("errorDescription", "")
                return f"ERROR: CapSolver poll errorId={poll_resp['errorId']}: {desc}"

            status = poll_resp.get("status")
            if status == "ready":
                solution = poll_resp.get("solution", {})
                # Different CAPTCHA types return the token under different keys.
                token = (
                    solution.get("gRecaptchaResponse")
                    or solution.get("token")
                    or solution.get("fcToken")
                )
                if token:
                    logger.info(
                        "CapSolver solved %s CAPTCHA (taskId=%s, attempt=%d)",
                        captcha_type,
                        task_id,
                        attempt,
                    )
                    return token
                return f"ERROR: CapSolver returned ready but no token in solution: {solution}"

            # status == "processing" — keep polling

        return f"ERROR: CapSolver timed out after {MAX_POLLS * 3}s for taskId={task_id}"

    return tools
