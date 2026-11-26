#!/usr/bin/env python3
"""Frankengine Parallel Builder (Skeleton-of-Thought + Shared-Prefix GPU clone_slot).

Turns ONE single user prompt into a full multi-module playable Three.js/WebGL app
using a 2-stage parallel architecture:
  Stage 1 (Architect, c=1): Generates a structured N-module Blueprint with fixed contracts.
  Stage 2 (N-Lane Fan-Out, clone_slot): Launches all N lanes simultaneously sharing the
          exact same Blueprint prefix (prefilled ONCE on GPU, cloned in <1ms/lane),
          streaming ~12K-20K+ tokens of code in parallel and stitching them into ONE
          self-contained playable HTML file.

Zero external dependencies (Python stdlib only). Works with http:// and https:// endpoints.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
from pathlib import Path


def stream_chat(
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    effort: str,
    enable_thinking: bool = False,
    timeout: int = 180,
) -> dict:
    parsed = urllib.parse.urlparse(url.rstrip("/") + "/chat/completions")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if enable_thinking:
        payload["reasoning_effort"] = effort
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(4):
        t0 = time.perf_counter()
        t_first = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = 0
        cached_tokens = 0
        completion_tokens = 0

        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(
                parsed.hostname, parsed.port or 443, timeout=timeout, context=ssl.create_default_context()
            )
        else:
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)

        try:
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status in (500, 502, 503, 504) and attempt < 3:
                resp.read()
                last_err = RuntimeError(f"HTTP {resp.status}")
                time.sleep(0.15 * (attempt + 1))
                continue
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:240]!r}")
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                ev = json.loads(data)
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    c = delta.get("content") or ""
                    r = delta.get("reasoning_content") or ""
                    if (c or r) and t_first is None:
                        t_first = time.perf_counter()
                    if c:
                        content_parts.append(c)
                    if r:
                        reasoning_parts.append(r)
                usage = ev.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            t_end = time.perf_counter()
            ttft = (t_first - t0) if t_first else (t_end - t0)
            dec_s = max(1e-6, t_end - (t_first or t0))
            return {
                "content": "".join(content_parts),
                "reasoning": "".join(reasoning_parts),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "ttft_s": round(ttft, 3),
                "total_s": round(t_end - t0, 3),
                "decode_tps": round(max(0, completion_tokens - 1) / dec_s, 1),
            }
        except (OSError, http.client.HTTPException) as e:
            last_err = e
            if attempt < 3:
                time.sleep(0.15 * (attempt + 1))
                continue
            raise
        finally:
            conn.close()
    raise last_err or RuntimeError("request failed")


def extract_json(text: str) -> dict:
    blocks = re.findall(r"```(?:json|jsonl)?\s*\n(.*?)```", text, flags=re.S)
    cand = blocks[-1] if blocks else text
    start = cand.find("{")
    end = cand.rfind("}")
    if start >= 0 and end > start:
        raw = cand[start : end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    # Resilient fallback: extract per-module objects line by line or via regex
    modules = []
    for line in cand.splitlines():
        line = line.strip().rstrip(",")
        if '"id"' in line and line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if "id" in obj and "spec" in obj:
                    modules.append(obj)
            except json.JSONDecodeError:
                m_id = re.search(r'"id"\s*:\s*"([^"]+)"', line)
                m_ti = re.search(r'"title"\s*:\s*"([^"]+)"', line)
                m_zo = re.search(r'"zone"\s*:\s*"([^"]*)"', line)
                m_sp = re.search(r'"spec"\s*:\s*"(.*)"\s*\}$', line)
                if m_id and m_sp:
                    modules.append({
                        "id": m_id.group(1),
                        "title": m_ti.group(1) if m_ti else m_id.group(1),
                        "zone": m_zo.group(1) if m_zo else "island",
                        "spec": m_sp.group(1).replace('"', "'"),
                    })
    if modules:
        m_title = re.search(r'"title"\s*:\s*"([^"]+)"', cand[:400])
        return {
            "title": m_title.group(1) if m_title else "Wuhu Island PS1 — Parallel Build",
            "modules": modules,
        }
    raise ValueError("Could not parse blueprint from architect response")


def extract_js(text: str) -> str:
    blocks = re.findall(r"```(?:javascript|js)?\s*\n(.*?)```", text, flags=re.S)
    code = blocks[-1].strip() if blocks else text.strip()
    return code


SHARED_RUNTIME_CONTRACT = """
SHARED THREE.JS RUNTIME CONTRACT (`ctx` passed into every module):
- `THREE`: global Three.js r128 library (already loaded).
- `ctx.scene`: THREE.Scene (sunny tropical blue sky, soft warm fog, hemisphere + directional sun light).
- `ctx.camera`: THREE.PerspectiveCamera (starts overlooking the island from (0, 58, 125)).
- `ctx.islandRadius`: 95 (world units, centered at (0, 0, 0)).
- `ctx.seaLevel`: 0.0.
- `ctx.getHeight(x, z)`: deterministic terrain height function returning Y >= -12 (0 at shoreline, peaks up to ~48 at the central volcano at (0, 0, -10), rolling green hills elsewhere, flat beach ring around radius 72..88).
- `ctx.ps1Mat(hexColor, opts)`: returns a flat-shaded low-poly PS1 MeshLambertMaterial / MeshStandardMaterial with `flatShading: true` and subtle vertex-snap style (opts may include `emissive`, `transparent`, `opacity`, `side`, `roughness`).
- `ctx.addUpdater(fn)`: register `fn(dt, elapsed)` called every animation frame (use for waves, spinning windmills, flying planes, boats, seagulls, waterfalls, particles, clouds).
- `ctx.addLandmark(name, posVec3, lookAtVec3)`: register a clickable camera teleport bookmark in the HUD.
- `ctx.makeGroup(name)`: creates a THREE.Group, adds it to `ctx.scene`, and returns it.
- Coordinate guide so all modules fit cleanly on the island without colliding awkwardly:
  * Central Volcano (Maka Wuhu): centered at (0, -10), base radius ~34, summit crater Y ~48.
  * Sandy Beach & Resort Town Harbor (South): z = +55..+82, x = -30..+30.
  * Lighthouse Cape (Southwest peninsula): (-62, +48).
  * Red Iron Arch Bridge & Lagoon (Southeast): (+46, +42) spanning a turquoise inlet.
  * Castle Ruins & Hilltop Plateau (Northwest): (-48, -36).
  * Waterfall & Mountain Lake (East slope): (+34, -18).
  * Golf Course & Windmills (West rolling hills): (-54, 0).
  * Swordplay Colosseum & Sports Courts (North plateau): (+12, -58).
  * Ocean & Distant Islets / Boats / Yachts: radius 96..210.
  * Sky, Clouds, Biplanes, Seagulls, Hot-Air Balloons: Y = 38..110.

RULES FOR EVERY LANE MODULE:
1. Register your module ONLY via:
   `window.WUHU_MODULES["<module_id>"] = function(ctx) { ... };`
2. Return ONLY pure JavaScript inside a single ```javascript``` code block. No HTML, no imports/exports.
3. Use ONLY standard Three.js r128 geometries (`BoxGeometry`, `ConeGeometry`, `CylinderGeometry`, `SphereGeometry`, `PlaneGeometry`, `TorusGeometry`, `DodecahedronGeometry`, `IcosahedronGeometry`, `BufferGeometry`, `LatheGeometry`, `RingGeometry`). Do NOT use `CapsuleGeometry` (not in r128!).
4. Keep geometry low-poly and PS1-authentic (4-10 radial segments, vivid sunny retro colors, flat shading via `ctx.ps1Mat`), rich in recognizable visual detail (25-80 meshes/instances per module), and always place ground objects accurately using `ctx.getHeight(x, z)`.
5. Never throw exceptions; keep code self-contained and clean.
""".strip()


def run_architect(url: str, model: str, user_prompt: str, n_lanes: int) -> tuple[dict, dict]:
    sys_msg = (
        "You are a Principal 3D Graphics Architect designing a rich, vibrant low-poly PS1-style Three.js scene "
        "that will be built in parallel by independent specialist worker lanes."
    )
    prompt = (
        f"User request: {user_prompt}\n\n"
        f"{SHARED_RUNTIME_CONTRACT}\n\n"
        f"Design a cohesive Blueprint with EXACTLY {n_lanes} distinct, non-overlapping visual & interactive modules "
        f"that together create an unforgettable sunny low-poly PS1 island world (terrain, ocean, volcano, bridges, "
        f"town, harbor, lighthouse, ruins, waterfall, forests, vehicles, wildlife, sky, atmosphere, sports venues, "
        f"boats, piers, props, lighting accents, mini-effects, etc.).\n\n"
        f"Return ONLY valid JSON inside a ```json``` block with this exact schema (put each module on ONE single line, never use unescaped double quotes inside strings):\n"
        f'{{\n'
        f'  "title": "Short catchy title",\n'
        f'  "palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5"],\n'
        f'  "style_notes": "2 concise sentences on the PS1 sunny aesthetic",\n'
        f'  "modules": [\n'
        f'    {{"id": "01_volcano", "title": "Maka Wuhu Volcano", "zone": "(0, -10)", "spec": "Detailed 1-2 sentence visual and animation spec"}}\n'
        f'  ]\n'
        f'}}\n'
        f"Remember: EXACTLY {n_lanes} one-line entries in `modules`, with unique snake_case `id`s."
    )
    res = stream_chat(
        url=url,
        model=model,
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
        max_tokens=3200,
        temperature=0.0,
        effort="low",
        enable_thinking=False,
    )
    bp = extract_json(res["content"])
    mods = bp.get("modules") or []
    if len(mods) < n_lanes:
        for i in range(len(mods), n_lanes):
            mods.append(
                {
                    "id": f"{i+1:02d}_island_detail_{i+1}",
                    "title": f"Island Scenic Detail #{i+1}",
                    "zone": f"({-60 + (i*19)%120}, {-50 + (i*23)%100})",
                    "spec": "Add charming low-poly PS1 island props (benches, signposts, flowers, rocks, lanterns, or boats) placed neatly on terrain with gentle motion.",
                }
            )
    bp["modules"] = mods[:n_lanes]
    return bp, res


def build_shared_prefix(user_prompt: str, blueprint: dict) -> str:
    bp_summary = json.dumps(blueprint, indent=1)
    return (
        f"PROJECT REQUEST: {user_prompt}\n\n"
        f"{SHARED_RUNTIME_CONTRACT}\n\n"
        f"MASTER BLUEPRINT (shared identically by all {len(blueprint['modules'])} parallel lanes):\n"
        f"{bp_summary}\n\n"
        "You are one specialist worker lane in a parallel code-generation swarm building this single world simultaneously.\n"
        "Follow the Shared Three.js Runtime Contract strictly so your module seamlessly composes with all sibling lanes.\n"
    )


def run_lane_worker(
    url: str,
    model: str,
    shared_sys: str,
    shared_prefix: str,
    idx: int,
    mod: dict,
    max_tokens: int,
    temperature: float,
    effort: str,
) -> dict:
    lane_suffix = (
        f"\n--- YOUR ASSIGNED LANE #{idx+1:02d} ---\n"
        f"Module ID: {mod['id']}\n"
        f"Title: {mod['title']}\n"
        f"Zone: {mod.get('zone', 'island')}\n"
        f"Spec: {mod['spec']}\n\n"
        f"IMPORTANT: Do NOT output JSON and do NOT repeat the spec above.\n"
        f"Output ONLY raw Three.js r128 JavaScript inside a single ```javascript``` code block that begins with:\n"
        f"window.WUHU_MODULES[\"{mod['id']}\"] = function(ctx) {{\n"
        f"Build rich low-poly PS1 geometry using ctx.ps1Mat, ctx.getHeight, ctx.makeGroup, ctx.addLandmark, and ctx.addUpdater."
    )
    # Slight stagger (15ms/lane = 0.45s total) keeps reverse-proxy forward-auth happy while all 32 arrive before leader prefill finishes.
    if idx > 0 and url.startswith("https://"):
        time.sleep(idx * 0.015)
    res = stream_chat(
        url=url,
        model=model,
        messages=[
            {"role": "system", "content": shared_sys},
            {"role": "user", "content": shared_prefix + lane_suffix},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        effort=effort,
        enable_thinking=False,
    )
    code = extract_js(res["content"])
    return {
        "lane": idx,
        "id": mod["id"],
        "title": mod["title"],
        "zone": mod.get("zone", ""),
        "code": code,
        "prompt_tokens": res["prompt_tokens"],
        "cached_tokens": res["cached_tokens"],
        "completion_tokens": res["completion_tokens"],
        "ttft_s": res["ttft_s"],
        "total_s": res["total_s"],
        "decode_tps": res["decode_tps"],
    }


def stitch_playable_html(blueprint: dict, lanes: list[dict], telemetry: dict) -> str:
    title = blueprint.get("title") or "Wuhu Island PS1 — Frankengine Parallel Build"
    module_scripts = []
    for ln in lanes:
        safe_code = ln["code"].replace("</script>", "<\\/script>")
        module_scripts.append(
            f"// === LANE {ln['lane']+1:02d}: {ln['id']} ({ln['completion_tokens']} tok, cached={ln['cached_tokens']}, {ln['ttft_s']}s TTFT) ===\n"
            f"try {{\n{safe_code}\n}} catch (e) {{ console.warn('Define failed {ln['id']}:', e); }}\n"
        )
    all_modules_js = "\n".join(module_scripts)
    tel_badge = (
        f"Built in parallel by {telemetry['lanes']} Frankengine lanes · "
        f"{telemetry['total_tokens']:,} tok in {telemetry['total_wall_s']:.2f}s "
        f"({telemetry['fanout_wall_tps']:.0f} tok/s wall · {telemetry['cloned_lanes']}/{telemetry['lanes']} clone_slot)"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #4da6ff; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  #gl {{ width: 100%; height: 100%; display: block; image-rendering: pixelated; }}
  #hud {{
    position: fixed; top: 12px; left: 12px; z-index: 10;
    background: rgba(12, 18, 30, 0.78); color: #eef6ff;
    border: 2px solid rgba(255, 232, 128, 0.75); border-radius: 6px;
    padding: 10px 12px; max-width: 420px; font-size: 12px; line-height: 1.4;
    backdrop-filter: blur(4px);
  }}
  #hud h1 {{ margin: 0 0 4px 0; font-size: 14px; color: #ffe066; letter-spacing: 0.03em; }}
  #hud .stats {{ color: #7ee787; font-size: 11px; margin-bottom: 6px; }}
  #hud .controls {{ color: #b8d0eb; font-size: 11px; }}
  #hud .bar {{ display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }}
  #hud button {{
    background: #1f324d; color: #eef6ff; border: 1px solid #58a6ff;
    border-radius: 4px; padding: 3px 7px; font: inherit; font-size: 11px; cursor: pointer;
  }}
  #hud button:hover {{ background: #2d4c75; }}
  #landmarks {{
    position: fixed; bottom: 12px; left: 12px; right: 12px; z-index: 10;
    display: flex; flex-wrap: wrap; gap: 5px; pointer-events: none;
  }}
  #landmarks button {{
    pointer-events: auto;
    background: rgba(12, 18, 30, 0.72); color: #ffe066;
    border: 1px solid rgba(255, 224, 102, 0.45); border-radius: 4px;
    padding: 4px 8px; font: inherit; font-size: 11px; cursor: pointer;
  }}
  #landmarks button:hover {{ background: rgba(35, 58, 92, 0.92); }}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="hud">
  <h1>{title}</h1>
  <div class="stats">{tel_badge}</div>
  <div class="controls">Drag: Look · WASD / Arrows: Fly · Space/Shift: Up/Down · R: Orbit</div>
  <div class="bar">
    <button id="btnPs1">PS1 320×240: ON</button>
    <button id="btnOrbit">Auto-Orbit: ON</button>
    <button id="btnStatus">Modules: loading...</button>
  </div>
</div>
<div id="landmarks"></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script>
window.WUHU_MODULES = {{}};
(function() {{
  const canvas = document.getElementById('gl');
  const renderer = new THREE.WebGLRenderer({{ canvas, antialias: false, powerPreference: 'high-performance' }});
  renderer.shadowMap.enabled = false;
  renderer.setClearColor(0x6ec6ff, 1);

  let ps1LowRes = true;
  function updateSize() {{
    const w = window.innerWidth, h = window.innerHeight;
    if (ps1LowRes) {{
      const scale = Math.max(1, Math.floor(Math.min(w / 420, h / 280)));
      renderer.setPixelRatio(1);
      renderer.setSize(Math.floor(w / scale), Math.floor(h / scale), false);
    }} else {{
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(w, h, false);
    }}
  }}
  updateSize();
  window.addEventListener('resize', updateSize);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x69c4ff);
  scene.fog = new THREE.Fog(0x9bdcff, 145, 420);

  const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.5, 900);
  camera.position.set(0, 62, 138);
  camera.lookAt(0, 14, -6);

  // Sunny Wii Sports Resort daylight
  const hemi = new THREE.HemisphereLight(0xd9f4ff, 0x3b8c52, 0.95);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff4cf, 1.15);
  sun.position.set(110, 140, 85);
  scene.add(sun);

  // Deterministic Wuhu-inspired heightmap
  function getHeight(x, z) {{
    const r = Math.hypot(x, z);
    const angle = Math.atan2(z, x);
    // Slightly organic coastline with southern bay & SE lagoon
    const coastMod = 1.0 + 0.11 * Math.sin(angle * 3.0 + 0.5) + 0.07 * Math.cos(angle * 5.0 - 0.8);
    const effR = r / (92.0 * coastMod);
    if (effR > 1.25) return -12.0;

    // Base island shelf
    let h = Math.max(-10.0, (1.0 - effR * effR) * 18.0);
    // Gently undulating hills
    h += Math.sin(x * 0.055) * Math.cos(z * 0.055) * 4.2 * Math.max(0, 1.0 - effR);
    h += Math.sin(x * 0.11 + 1.3) * Math.sin(z * 0.09) * 1.8 * Math.max(0, 1.0 - effR);

    // Central Maka Wuhu volcano at (0, -10)
    const vr = Math.hypot(x - 0, z + 10);
    if (vr < 42) {{
      const t = 1.0 - vr / 42.0;
      let peak = t * t * 42.0;
      if (vr < 8.5) peak -= (1.0 - vr / 8.5) * 9.0; // caldera crater
      h += peak;
    }}
    // NW castle plateau (-48, -36)
    const cr = Math.hypot(x + 48, z + 36);
    if (cr < 20) h = Math.max(h, 14.0 * (1.0 - cr / 22.0) + 5.0);

    // SE lagoon cut (+46, +42)
    const lr = Math.hypot(x - 46, z - 42);
    if (lr < 16) h -= (1.0 - lr / 16.0) * 11.0;

    // South beach flattening
    if (z > 55 && Math.abs(x) < 34 && h > 0.2 && h < 5.5) h *= 0.45;
    return h;
  }}

  // Base fallback island terrain + ocean (enhanced by modules)
  const geo = new THREE.PlaneGeometry(260, 260, 56, 56);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const cols = [];
  const cSand = new THREE.Color(0xf7df94);
  const cGrass = new THREE.Color(0x56c24a);
  const cHill = new THREE.Color(0x3ba03d);
  const cRock = new THREE.Color(0x7d746d);
  const cPeak = new THREE.Color(0x5b534f);
  const cBed = new THREE.Color(0x2d94b8);
  for (let i = 0; i < pos.count; i++) {{
    const x = pos.getX(i), z = pos.getZ(i);
    const y = getHeight(x, z);
    pos.setY(i, y);
    let c = cGrass;
    if (y < -0.5) c = cBed;
    else if (y < 2.2) c = cSand;
    else if (y < 14) c = cGrass;
    else if (y < 26) c = cHill;
    else if (y < 38) c = cRock;
    else c = cPeak;
    cols.push(c.r, c.g, c.b);
  }}
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  const baseTerrain = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({{ vertexColors: true, flatShading: true }}));
  scene.add(baseTerrain);

  const seaGeo = new THREE.PlaneGeometry(900, 900, 32, 32);
  seaGeo.rotateX(-Math.PI / 2);
  const baseSea = new THREE.Mesh(seaGeo, new THREE.MeshLambertMaterial({{ color: 0x1ca3ec, transparent: true, opacity: 0.86, flatShading: true }}));
  baseSea.position.y = 0;
  scene.add(baseSea);

  function ps1Mat(hexColor, opts) {{
    const o = Object.assign({{ color: hexColor, flatShading: true }}, opts || {{}});
    return new THREE.MeshLambertMaterial(o);
  }}

  const updaters = [];
  const landmarks = [
    {{ name: 'Overview', pos: new THREE.Vector3(0, 62, 138), look: new THREE.Vector3(0, 14, -6) }},
    {{ name: 'Volcano', pos: new THREE.Vector3(26, 56, 22), look: new THREE.Vector3(0, 38, -10) }},
    {{ name: 'Harbor', pos: new THREE.Vector3(0, 18, 102), look: new THREE.Vector3(0, 6, 62) }},
  ];

  const ctx = {{
    THREE, scene, camera, renderer,
    islandRadius: 95, seaLevel: 0.0,
    getHeight, ps1Mat,
    addUpdater: (fn) => {{ if (typeof fn === 'function') updaters.push(fn); }},
    addLandmark: (name, p, l) => {{
      if (name && p && l && landmarks.length < 18) landmarks.push({{ name, pos: p.clone ? p.clone() : new THREE.Vector3(p.x||0, p.y||20, p.z||50), look: l.clone ? l.clone() : new THREE.Vector3(l.x||0, l.y||5, l.z||0) }});
    }},
    makeGroup: (name) => {{ const g = new THREE.Group(); g.name = name || 'mod'; scene.add(g); return g; }},
  }};

  window.__WUHU_CTX__ = ctx;
  window.__WUHU_RUN__ = function() {{
    let ok = 0, fail = 0;
    for (const [id, fn] of Object.entries(window.WUHU_MODULES)) {{
      try {{ fn(ctx); ok++; }} catch (e) {{ fail++; console.warn('Module runtime error:', id, e); }}
    }}
    document.getElementById('btnStatus').textContent = `Modules: ${{ok}}/${{ok+fail}} active`;
    window.__WUHU_STATS__ = {{ ok, fail, meshes: scene.children.length }};

    const lmDiv = document.getElementById('landmarks');
    lmDiv.innerHTML = '';
    landmarks.forEach((lm) => {{
      const b = document.createElement('button');
      b.textContent = lm.name;
      b.onclick = () => {{
        autoOrbit = false;
        document.getElementById('btnOrbit').textContent = 'Auto-Orbit: OFF';
        camera.position.copy(lm.pos);
        camera.lookAt(lm.look);
        syncYawPitch();
      }};
      lmDiv.appendChild(b);
    }});
  }};

  // Camera fly + orbit controls
  let autoOrbit = true;
  let orbitAngle = 0.0;
  let yaw = 0, pitch = -0.35;
  function syncYawPitch() {{
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    yaw = Math.atan2(-dir.x, -dir.z);
    pitch = Math.asin(Math.max(-0.98, Math.min(0.98, dir.y)));
  }}
  syncYawPitch();

  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener('mousedown', (e) => {{ dragging = true; lx = e.clientX; ly = e.clientY; autoOrbit = false; document.getElementById('btnOrbit').textContent = 'Auto-Orbit: OFF'; }});
  window.addEventListener('mouseup', () => dragging = false);
  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    yaw -= (e.clientX - lx) * 0.005;
    pitch = Math.max(-1.35, Math.min(1.35, pitch - (e.clientY - ly) * 0.005));
    lx = e.clientX; ly = e.clientY;
    camera.rotation.order = 'YXZ';
    camera.rotation.y = yaw;
    camera.rotation.x = pitch;
  }});

  const keys = {{}};
  window.addEventListener('keydown', (e) => {{
    keys[e.code] = true;
    if (e.code === 'KeyR') {{
      autoOrbit = !autoOrbit;
      document.getElementById('btnOrbit').textContent = 'Auto-Orbit: ' + (autoOrbit ? 'ON' : 'OFF');
    }}
  }});
  window.addEventListener('keyup', (e) => keys[e.code] = false);

  document.getElementById('btnPs1').onclick = () => {{
    ps1LowRes = !ps1LowRes;
    document.getElementById('btnPs1').textContent = 'PS1 320×240: ' + (ps1LowRes ? 'ON' : 'OFF');
    updateSize();
  }};
  document.getElementById('btnOrbit').onclick = () => {{
    autoOrbit = !autoOrbit;
    document.getElementById('btnOrbit').textContent = 'Auto-Orbit: ' + (autoOrbit ? 'ON' : 'OFF');
  }};

  const clock = new THREE.Clock();
  function animate() {{
    requestAnimationFrame(animate);
    const dt = Math.min(0.05, clock.getDelta());
    const elapsed = clock.elapsedTime;
    if (autoOrbit) {{
      orbitAngle += dt * 0.14;
      camera.position.set(Math.sin(orbitAngle) * 135, 56 + Math.sin(orbitAngle * 2) * 8, Math.cos(orbitAngle) * 135);
      camera.lookAt(0, 14, -6);
    }} else {{
      const speed = 42 * dt;
      const fwd = new THREE.Vector3();
      camera.getWorldDirection(fwd);
      const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
      if (keys['KeyW'] || keys['ArrowUp']) camera.position.addScaledVector(fwd, speed);
      if (keys['KeyS'] || keys['ArrowDown']) camera.position.addScaledVector(fwd, -speed);
      if (keys['KeyD'] || keys['ArrowRight']) camera.position.addScaledVector(right, speed);
      if (keys['KeyA'] || keys['ArrowLeft']) camera.position.addScaledVector(right, -speed);
      if (keys['Space']) camera.position.y += speed;
      if (keys['ShiftLeft'] || keys['ShiftRight']) camera.position.y -= speed;
    }}
    baseSea.position.y = Math.sin(elapsed * 1.3) * 0.18;
    for (let i = 0; i < updaters.length; i++) {{
      try {{ updaters[i](dt, elapsed); }} catch (_) {{}}
    }}
    renderer.render(scene, camera);
  }}
  window.__WUHU_START__ = animate;
}})();
</script>
<script>
{all_modules_js}
window.__WUHU_RUN__();
window.__WUHU_START__();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Frankengine Parallel Builder (Architect + N-Lane clone_slot Fan-Out)")
    ap.add_argument("--url", default=os.environ.get("FRANKENGINE_URL", "http://127.0.0.1:8834/v1"))
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument(
        "--prompt",
        default="build me a sunny low-poly island like Wuhu Island from Wii Sports Resort, with PS1 graphics",
    )
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=520)
    ap.add_argument("--temperature", type=float, default=0.55)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out-html", default="wuhu_island_ps1.html")
    ap.add_argument("--out-json", default="wuhu_island_telemetry.json")
    args = ap.parse_args()

    t_total0 = time.perf_counter()
    print(f"[Stage 1/3] Architecting {args.lanes}-module Blueprint (solo c=1)...", flush=True)
    bp, arch_res = run_architect(args.url, args.model, args.prompt, args.lanes)
    print(
        f"  Blueprint ready: '{bp.get('title')}' ({len(bp['modules'])} modules) in {arch_res['total_s']}s "
        f"({arch_res['completion_tokens']} tok @ {arch_res['decode_tps']} tok/s)\n",
        flush=True,
    )

    shared_sys = (
        "You are an expert Three.js r128 low-poly PS1 3D graphics engineer. "
        "Output ONLY clean, bug-free JavaScript inside a ```javascript``` block."
    )
    shared_prefix = build_shared_prefix(args.prompt, bp)

    print(
        f"[Stage 2/3] Launching ALL {len(bp['modules'])} parallel lanes simultaneously (shared-prefix clone_slot)...",
        flush=True,
    )
    t_fan0 = time.perf_counter()
    lanes_out: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bp["modules"])) as ex:
        futs = [
            ex.submit(
                run_lane_worker,
                args.url,
                args.model,
                shared_sys,
                shared_prefix,
                i,
                mod,
                args.max_tokens,
                args.temperature,
                args.effort,
            )
            for i, mod in enumerate(bp["modules"])
        ]
        done_n = 0
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            lanes_out.append(r)
            done_n += 1
            print(
                f"  [{done_n:02d}/{len(bp['modules'])}] {r['id']:26s} | cached={r['cached_tokens']:5d} | "
                f"ttft={r['ttft_s']:5.2f}s | out={r['completion_tokens']:4d}t ({r['decode_tps']:5.1f} t/s)",
                flush=True,
            )
    fan_wall = time.perf_counter() - t_fan0
    lanes_out.sort(key=lambda x: x["lane"])
    total_wall = time.perf_counter() - t_total0

    fan_tokens = sum(x["completion_tokens"] for x in lanes_out)
    total_tokens = arch_res["completion_tokens"] + fan_tokens
    total_prompt = sum(x["prompt_tokens"] for x in lanes_out)
    total_cached = sum(x["cached_tokens"] for x in lanes_out)
    cloned_lanes = sum(1 for x in lanes_out if x["cached_tokens"] >= 64)
    ttfts = sorted(x["ttft_s"] for x in lanes_out)
    sum_decode_tps = sum(x["decode_tps"] for x in lanes_out)
    fan_wall_tps = fan_tokens / max(1e-6, fan_wall)
    total_wall_tps = total_tokens / max(1e-6, total_wall)

    telemetry = {
        "prompt": args.prompt,
        "title": bp.get("title"),
        "lanes": len(lanes_out),
        "architect_s": arch_res["total_s"],
        "architect_tokens": arch_res["completion_tokens"],
        "architect_tps": arch_res["decode_tps"],
        "fanout_wall_s": round(fan_wall, 2),
        "fanout_tokens": fan_tokens,
        "fanout_wall_tps": round(fan_wall_tps, 1),
        "fanout_sum_decode_tps": round(sum_decode_tps, 1),
        "total_wall_s": round(total_wall, 2),
        "total_tokens": total_tokens,
        "total_wall_tps": round(total_wall_tps, 1),
        "logical_prompt_tokens": total_prompt,
        "cached_prompt_tokens": total_cached,
        "cache_hit_pct": round(100.0 * total_cached / max(1, total_prompt), 1),
        "cloned_lanes": cloned_lanes,
        "ttft_min_s": ttfts[0],
        "ttft_p50_s": ttfts[len(ttfts) // 2],
        "ttft_max_s": ttfts[-1],
        "est_linear_solo_s": round(total_tokens / 150.0, 1),
        "speedup_vs_solo": round((total_tokens / 150.0) / max(1e-6, total_wall), 2),
        "lanes_data": [{k: v for k, v in x.items() if k != "code"} for x in lanes_out],
    }

    html = stitch_playable_html(bp, lanes_out, telemetry)
    out_html = Path(args.out_html)
    out_html.write_text(html, encoding="utf-8")
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")

    print(
        f"\n[Stage 3/3] STITCHED PLAYABLE APP -> {out_html.resolve()}\n"
        f"  Lanes cloned     : {cloned_lanes}/{telemetry['lanes']} ({telemetry['cache_hit_pct']}% of {total_prompt:,} logical prompt tok)\n"
        f"  Fan-out TTFT     : min={telemetry['ttft_min_s']}s / p50={telemetry['ttft_p50_s']}s / max={telemetry['ttft_max_s']}s\n"
        f"  Fan-out decode   : {fan_tokens:,} tok in {fan_wall:.2f}s ({fan_wall_tps:.1f} tok/s wall | {sum_decode_tps:.1f} tok/s sum)\n"
        f"  Total E2E        : {total_tokens:,} tok in {total_wall:.2f}s ({total_wall_tps:.1f} tok/s wall)\n"
        f"  vs Solo c=1 (~150 t/s): ~{telemetry['est_linear_solo_s']}s linear -> {telemetry['speedup_vs_solo']}x faster!\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
