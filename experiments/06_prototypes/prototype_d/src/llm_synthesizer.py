"""
prototype_d/src/llm_synthesizer.py

LLM-Based Fallback Synthesizer — Cerebras Backend
====================================================
Uses Cerebras Cloud API (llama3.1-8b) via its OpenAI-compatible REST endpoint.
No SDK required — pure urllib.

API key: set CEREBRAS_API_KEY env variable OR pass api_key= directly.

Usage:
    from llm_synthesizer import llm_synthesize_rule
    rule_fn, description = llm_synthesize_rule(demo_pairs, test_input)
"""

from __future__ import annotations

import ast
import json
import os
import textwrap
import urllib.request
import urllib.error
from typing import List, Optional, Tuple

Grid = List[List[int]]
DemoPair = Tuple[Grid, Grid]

# ── Cerebras config ───────────────────────────────────────────────────────────
CEREBRAS_API_KEY = os.environ.get(
    "CEREBRAS_API_KEY",
    "csk-km3rxy8emy4p35kfc95t8tk95vxcy9r8f8pftkhxh8yp39x5"
)
CEREBRAS_URL    = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL  = "llama3.1-8b"   # fast, cheap — upgrade to llama3.3-70b for harder tasks


# ─── PROMPT BUILDING ──────────────────────────────────────────────────────────

def _grid_to_str(g: Grid) -> str:
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in g) + "]"


def _build_prompt(demo_pairs: List[DemoPair], test_input: Grid) -> str:
    examples_text = ""
    for i, (inp, out) in enumerate(demo_pairs):
        examples_text += (
            f"\nExample {i+1}:\n"
            f"  input  = {_grid_to_str(inp)}\n"
            f"  output = {_grid_to_str(out)}\n"
        )

    return textwrap.dedent(f"""
    You are solving an ARC-AGI task. Given the examples below, write a Python function
    `transform(grid)` that maps every input to its correct output.

    Rules:
    - grid is List[List[int]] with values 0-9
    - Your function must work for ALL examples shown
    - Look for the single transformation rule
    - Return ONLY the Python function, nothing else — no explanation, no imports, no markdown fences

    {examples_text}
    Test input (your function will be applied to this):
    {_grid_to_str(test_input)}

    Write ONLY the Python function `transform(grid)`:
    """).strip()


# ─── CEREBRAS API CALL ────────────────────────────────────────────────────────

def _call_cerebras(prompt: str, api_key: str = CEREBRAS_API_KEY) -> str:
    """Call Cerebras API and return raw text response."""
    payload = json.dumps({
        "model": CEREBRAS_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 512,
        "temperature": 0.2,
        "top_p": 1,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        CEREBRAS_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# ─── SAFE EXEC & VERIFY ───────────────────────────────────────────────────────

def _safe_exec_transform(
    code: str, demo_pairs: List[DemoPair]
) -> Tuple[bool, Optional[object]]:
    """
    Exec the synthesized code in a sandbox and verify it against all demos.
    Returns (is_valid, transform_fn_or_None).
    """
    # Strip markdown fences if the model forgot
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    # Basic AST safety check — no imports, no file I/O
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, None

    forbidden_nodes = {"Import", "ImportFrom", "Global", "Nonlocal"}
    forbidden_calls = {"open", "exec", "eval", "__import__", "compile",
                       "subprocess", "os", "sys"}
    for node in ast.walk(tree):
        if type(node).__name__ in forbidden_nodes:
            return False, None
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls:
                return False, None

    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102
    except Exception:
        return False, None

    transform = namespace.get("transform")
    if not callable(transform):
        return False, None

    from primitives import grid_equal
    for inp, out in demo_pairs:
        try:
            predicted = transform(inp)
        except Exception:
            return False, None
        if not grid_equal(predicted, out):
            return False, None

    return True, transform


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def llm_synthesize_rule(
    demo_pairs: List[DemoPair],
    test_input: Grid,
    n_retries: int = 2,
    verbose: bool = False,
    api_key: str = CEREBRAS_API_KEY,
) -> Tuple[Optional[object], str]:
    """
    Ask Cerebras (llama3.1-8b) to synthesize a transform function from demos.

    Returns:
        (transform_fn, description)  — transform_fn is None if synthesis failed.
    """
    prompt = _build_prompt(demo_pairs, test_input)

    for attempt in range(1, n_retries + 1):
        try:
            code = _call_cerebras(prompt, api_key=api_key)

            if verbose:
                print(f"    [llm] attempt {attempt} — {len(code)} chars received")

            is_valid, fn = _safe_exec_transform(code, demo_pairs)
            if is_valid:
                if verbose:
                    print(f"    [llm] ✓ function verified on all demos")
                return fn, f"cerebras_{CEREBRAS_MODEL}_attempt{attempt}"
            else:
                if verbose:
                    print(f"    [llm] code failed demo verification (attempt {attempt})")

        except urllib.error.URLError as e:
            if verbose:
                print(f"    [llm] network error (attempt {attempt}): {e}")
            # Network blocked — fail fast instead of retrying
            break
        except Exception as e:
            if verbose:
                print(f"    [llm] error (attempt {attempt}): {e}")

    return None, "LLM_FAILED"
