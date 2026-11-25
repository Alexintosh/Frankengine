import collections
import json
import os
import urllib.request

base = os.environ.get("TP2_TEST_URL", "http://127.0.0.1:18000/v1")
model = json.load(urllib.request.urlopen(base + "/models", timeout=15))["data"][0]["id"]

def sample(seed):
    body = dict(model=model, temperature=.7, top_p=.9, top_k=40, seed=seed,
                max_tokens=1, enable_thinking=False,
                messages=[{"role": "user", "content": "Complete this sequence with any fitting color in one word: red, blue, yellow,"}])
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    result = json.load(urllib.request.urlopen(req, timeout=20))
    message = result["choices"][0]["message"]
    return (message.get("content") or message.get("reasoning_content") or "").strip()

outputs = [sample(seed) for seed in range(32)]
assert outputs == [sample(seed) for seed in range(32)]
print("32 seeds repeated exactly", dict(collections.Counter(outputs)), flush=True)
