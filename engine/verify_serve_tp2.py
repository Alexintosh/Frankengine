#!/usr/bin/env python3
import concurrent.futures
import json
import time
import urllib.request
import os
import re

BASE = os.environ.get("TP2_TEST_URL", "http://127.0.0.1:8000/v1")
model = json.load(urllib.request.urlopen(BASE + "/models", timeout=15))["data"][0]["id"]
print("model", model, flush=True)

def run(messages, *, temp=0, stream=True, limit=128, cancel=False, seed=42):
    body = dict(model=model, messages=messages, temperature=temp, max_tokens=limit, stream=stream, enable_thinking=False)
    if temp:
        body.update(top_p=.9, top_k=40, seed=seed)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    chunks = []
    events = 0
    usage = {}
    done = False
    with urllib.request.urlopen(req, timeout=90) as response:
        if not stream:
            result = json.load(response)
            return result["choices"][0]["message"]["content"], result["usage"]
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = line[6:].strip()
            if event == b"[DONE]":
                done = True
                break
            item = json.loads(event)
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice["delta"]
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    first = first or time.monotonic()
                    events += 1
                    if delta.get("content"):
                        chunks.append(delta["content"])
                    if cancel and events >= 20:
                        return "cancelled", events
    assert done and first is not None and usage, (chunks, usage)
    ttft = first - start
    speed = (usage["completion_tokens"] - 1) / max(time.monotonic() - first, .001)
    return "".join(chunks), dict(ttft=round(ttft, 3), tok_s=round(speed, 2), **usage)

def user(content):
    return [{"role": "user", "content": content}]

short = user("What is 2 + 2? Reply with just the number.")
plain = run(short, stream=False)
greedy = run(short)
assert plain[0] == greedy[0] and re.fullmatch(r"4(?:\s+4)*", plain[0].strip()), (plain, greedy)
print("greedy", greedy, flush=True)

def coding(label):
    lines = "\n".join(f"def fn_{i:03}(value): return value + {i}" for i in range(125))
    return user(f"{label}\nReview this Python module. Write a detailed refactor and test plan with code.\n{lines}")

for temp in (0, .7):
    messages = coding(f"Coding review {temp}")
    answer, metrics = run(messages, temp=temp, limit=300)
    print("cold", temp, metrics, repr(answer[:100]), flush=True)
    if temp:
        repeated, repeat_metrics = run(messages, temp=temp, limit=300)
        assert repeated == answer
        print("same seed exact repeat", repeat_metrics, flush=True)
        different, _ = run(messages, temp=temp, limit=300, seed=43)
        assert different != answer
        print("different seed diverges", flush=True)
    followup = messages + [{"role": "assistant", "content": answer}, *user("Now write the first two tests.")]
    follow_answer, follow_metrics = run(followup, temp=temp)
    print("follow-up", temp, follow_metrics, flush=True)
    run(coding("Unrelated review"), temp=.7 if not temp else 0, limit=20)
    parked_answer, parked_metrics = run(followup, temp=temp)
    assert parked_answer == follow_answer
    assert parked_metrics["ttft"] < metrics["ttft"] / 2
    print("parked across mode switch", temp, parked_metrics, flush=True)

assert run(coding("Cancellation"), temp=.7, limit=512, cancel=True)[0] == "cancelled"
run(short, temp=.7, limit=64)
assert run(short)[0] == plain[0]
print("cancel and immediate same-mode recovery", flush=True)

with concurrent.futures.ThreadPoolExecutor(2) as pool:
    outputs = list(pool.map(lambda _: run(short)[0], range(2)))
assert outputs == [plain[0], plain[0]], outputs
print("queued requests passed", flush=True)
