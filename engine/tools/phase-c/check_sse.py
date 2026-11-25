"""Record completed SSE responses and verify warm-cache and passkey gates."""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

base, tag, out, needles = sys.argv[1:]
model = json.load(urllib.request.urlopen(base + "/models", timeout=5))["data"][0]["id"]

def request(label, prompt, temperature=0, k=0, limit=160, reasoning_effort=None):
    body = dict(model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, top_p=.9, top_k=k, seed=42,
                max_tokens=limit, enable_thinking=reasoning_effort is not None, stream=True,
                stream_options={"include_usage": True})
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    pieces = []
    reasoning = []
    usage = {}
    done = False
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line[6:].strip() == b"[DONE]":
                done = True
                break
            item = json.loads(line[6:])
            usage = item.get("usage") or usage
            for choice in item["choices"]:
                delta = choice["delta"]
                for key, fragments in (("content", pieces), ("reasoning_content", reasoning)):
                    if delta.get(key):
                        first = first or time.monotonic()
                        fragments.append(delta[key])
    assert done and first and usage, (label, done, usage)
    result = dict(label=label, text="".join(pieces), reasoning="".join(reasoning),
                  ttft=first-start,
                  decode_tok_s=(usage["completion_tokens"]-1)/max(.001,time.monotonic()-first), usage=usage)
    with Path(out).open("a") as f:
        print(json.dumps(result), file=f)
    return result

request("short-12tok", "What is 2 + 2? Reply with just the number.", limit=12)
request("short-chat-160tok",
        "Explain how a hash map handles collisions, with a short Python example.",
        reasoning_effort="low")
failed = []
for temperature, k in ((0, 0), (.7, 40), (.7, 0)):
    lines = "\n".join(f"def fn_{i:03}(value): return value + {i}" for i in range(125))
    prompt = f"Review this Python module and write a detailed test plan. Case {temperature}/{k}.\n{lines}"
    cold = request(f"{temperature}/{k} cold", prompt, temperature, k)
    warm = request(f"{temperature}/{k} warm", prompt, temperature, k)
    cached = warm["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if (tag == "unified" and not cached) or cold["text"] != warm["text"]:
        failed.append((temperature, k, cached))
    if tag == "unified" and temperature == 0 and os.getenv("PHASE_C_SWITCH_ROUTE"):
        request("short route switch", "What is 3 + 3? Reply with just the number.", limit=12)
        restored = request("greedy restored after DFlash2", prompt)
        cached = restored["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        if not cached or cold["text"] != restored["text"]:
            failed.append(("greedy restored after DFlash2", cached))

if tag == "unified" and needles:
    for label, entry in json.loads(Path(needles).read_text()).items():
        result = request(label, entry["prompt"], limit=24)
        if result["text"].strip() != entry["key"]:
            failed.append((label, result["text"]))

if failed:
    raise SystemExit(str(failed))
