#!/usr/bin/env python3
"""Collect target-policy chat completions after checking against evaluation prompts."""

import argparse
import json
import math
import random
import re
import sys
import time
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DOMAINS = {
    "code": (
        (
            "a C++20 fixed-capacity ring of timestamped sensor samples",
            "a Rust iterator for records in a length-prefixed local file",
            "a Python library that merges sorted intervals from multiple readers",
            "a TypeScript reducer for asynchronous table pagination",
            "a PostgreSQL query pipeline that groups consecutive stockout days",
            "a C++20 scheduler for uploads with a bounded worker queue",
            "a Rust parser for temperature readings with units",
            "a Python formatter for nested progress reports",
            "a TypeScript cache with request coalescing and expiry",
            "a SQLite query pipeline for overlapping room reservations",
        ),
        (
            "Implement {topic}. State the public interface and include representative tests.",
            "Sketch a minimal version of {topic} and explain what each function is responsible for.",
            "Design cancellation or early termination for {topic}. Show how resources are released.",
            "For {topic}, write a small reference implementation and one example that runs end to end.",
            "Extend {topic} to handle malformed input and report record-level diagnostics.",
            "Design and implement fast and slow paths for {topic}. Justify the branching criteria.",
            "Show how to test boundary inputs for {topic} with deterministic fixtures and expected results.",
            "Refactor {topic} so the parsing or normalization is separate from state updates. Show both interfaces.",
            "Write a version of {topic} that can handle data larger than memory. Explain the memory bound.",
            "Build {topic} with resumable processing. Explain exactly what must be stored in a checkpoint.",
        ),
    ),
    "math": (
        (
            "counting paths through a grid with one blocked diagonal",
            "finding a minimum-cost ordering of intervals with precedence constraints",
            "bounding the error of a trapezoidal approximation to a convex function",
            "classifying connected components after batches of edge deletions",
            "computing hitting probabilities for a biased walk on a finite line",
            "finding the number of binary strings without runs of four ones",
            "maximizing the total value of compatible half-open intervals",
            "partitioning nonnegative weights into contiguous bounded groups",
            "finding when two periodic task sequences first coincide",
            "estimating variance for sampling a finite population without replacement",
        ),
        (
            "Develop a rigorous solution for {topic}. Work through one small example by hand.",
            "For {topic}, write a recurrence with base cases and prove it by induction.",
            "Explain the main mathematical idea behind {topic}, then show how to check a numeric result.",
            "Give a simple exhaustive method for {topic}. Describe how to use it as an oracle for small inputs.",
            "Analyze {topic} under two simplifying assumptions. Illustrate what changes when they are relaxed.",
            "Create a small contest problem about {topic} and provide a worked solution.",
            "Propose a scalable method for {topic}, with a proof sketch and a complexity analysis.",
            "Find a degenerate case for {topic} that invalidates a plausible shortcut. Derive a correct method.",
            "Explore how sensitive the result of {topic} is to changing one input parameter.",
            "Show two approaches to {topic}. Compare their intermediate quantities on the same example.",
        ),
    ),
    "reasoning": (
        (
            "a job queue that sometimes reports completion before its output file is durable",
            "a data pipeline that records the same daily totals twice after midnight",
            "a web report that momentarily shows an older filter after a user changes it",
            "a compiler cache where one test passes only after a clean build",
            "a build worker pool where one worker is always idle under load",
            "a shared calendar that assigns overlapping appointments across time zones",
            "a metrics dashboard whose totals differ between minute and hour views",
            "a migration where newly added rows are absent from historical reports",
            "a video import utility that gradually consumes memory when scanning directories",
            "a factory simulator whose output depends on the order in which events were queued",
        ),
        (
            "Diagnose {topic}. Rank three hypotheses and give the smallest experiment to distinguish them.",
            "Design a reproducible incident fixture for {topic}. Include an event timeline and checks.",
            "Write a debugging dialogue about {topic}: one investigator proposes a hypothesis and the other tests it.",
            "For {topic}, propose a state machine that rules out the reported behavior. Include recovery.",
            "Explain how to instrument {topic} using structured logs. Include sample event fields and queries.",
            "Write a rollback-aware repair plan for {topic}. State how to detect a safe moment to resume.",
            "Build a fault tree for {topic}, then explain how to eliminate branches without modifying live state.",
            "For {topic}, present two plausible causes with identical symptoms and a test that separates them.",
            "Turn {topic} into a postmortem with an explicit distinction between observations and hypotheses.",
            "Give a short design review of {topic}. Specify invariants and observable acceptance criteria.",
        ),
    ),
    "prose": (
        (
            "how a neighborhood library moves from manual checkouts to a local catalog",
            "connecting a workshop's temperature sensors to a local timeseries dashboard",
            "migrating a publishing team from shared folders to versioned manuscripts",
            "coordinating maintenance of a cooperative's bicycle lending fleet",
            "tracking calibrations for shared equipment in a teaching lab",
            "operating a local archive of digitized field notes",
            "handling delayed inventory feeds from multiple stores",
            "curating a long-lived dataset of city tree measurements",
            "introducing automated backups for a community radio production team",
            "maintaining a neighborhood garden's irrigation controller across seasons",
        ),
        (
            "Write a practical field guide on {topic} for a new team member.",
            "Prepare a project brief on {topic}. Include milestones and how each is verified.",
            "Write a release note on {topic} for people who rely on the existing workflow.",
            "Draft a two-page operations handbook about {topic}, including routine troubleshooting.",
            "Write a technical explainer on {topic} for a reader with domain knowledge and little software experience.",
            "Describe {topic} as a clear case study with concrete measurements to collect.",
            "Write a handover memo on {topic}, covering open decisions and the evidence needed to make them.",
            "Prepare a concise training exercise about {topic}, followed by a worked answer.",
            "Write a plain-language FAQ on {topic}. Use realistic examples and preserve technical accuracy.",
            "Compare two approaches to {topic} in the form of an architectural decision record.",
        ),
    ),
    "structured": (
        (
            "a hiking trail survey with routes, condition reports and photographs",
            "a batch order for a bakery with allergens and pickup times",
            "a bookstore intake ledger with partial shipments and edition identifiers",
            "a train timetable with service exceptions and interchange limits",
            "a small workshop's tool service log with recurring actions",
            "a library room booking with accessibility features and recurring intervals",
            "an orchard yield report with plot identifiers and unit conversions",
            "a theater rehearsal schedule with alternate casts and stage allocations",
            "a museum loan record with packing conditions and dated handoffs",
            "a data import policy with field mappings and retryable error categories",
        ),
        (
            "Define a JSON Schema for {topic} with required and optional fields and one valid instance.",
            "Represent {topic} as YAML with three realistic records and validation rules.",
            "Produce example JSON for {topic}. Include predictable identifiers and at least one missing optional value.",
            "Normalize {topic} into related tables and provide matching structured sample data.",
            "For {topic}, design two versions of a JSON document and show a deterministic migration between them.",
            "Write a JSON Schema for an event log covering {topic}, including discriminated event types.",
            "Design a YAML configuration for {topic}. Use defaults and explain how overrides merge.",
            "Describe {topic} in JSON Lines and include three records that exercise edge cases.",
            "Give a machine-readable export contract for {topic}, with annotated valid and invalid examples.",
            "Create a JSON test fixture for {topic} covering empty, ordinary and boundary-value cases.",
        ),
    ),
}

MULTILINGUAL = (
    (
        ("un orto scolastico con turni settimanali", "un archivio comunale di fotografie storiche",
         "una biblioteca che condivide strumenti musicali", "il controllo dei consumi in una palestra",
         "la gestione delle prenotazioni di un rifugio", "una piccola rete di stazioni meteo"),
        ("Scrivi una guida pratica per {topic} usando un esempio realistico.",
         "Definisci una procedura di verifica per {topic} e indica come registrare i risultati.",
         "Confronta due modi di organizzare {topic} in base ai costi e ai tempi.",
         "Prepara una tabella di lavoro per {topic} con responsabilità, scadenze e verifiche."),
    ),
    (
        ("un atelier qui prête des vélos", "un bureau qui numérise ses factures",
         "un réseau local de capteurs de bruit", "une médiathèque qui renouvelle son catalogue",
         "un laboratoire qui suit ses étalonnages", "un petit musée qui accueille des classes"),
        ("Rédige une notice claire pour {topic} et ajoute un exemple concret.",
         "Propose deux stratégies pour {topic} et explique comment évaluer leurs résultats.",
         "Prépare un scénario de formation pour {topic} avec des exercices pratiques.",
         "Élabore un plan de reprise pour {topic} après trois jours sans accès aux données."),
    ),
    (
        ("eine Werkstatt mit gemeinsam genutzten Maschinen", "ein Archiv mit historischen Tonaufnahmen",
         "eine Schule mit einem Leihlaptop-Programm", "ein Museum mit wechselnden Ausstellungen",
         "eine Kantine mit wöchentlichen Menüplänen", "ein Netz von Messstationen für Bodenfeuchte"),
        ("Schreibe eine verständliche Anleitung für {topic} und verwende ein konkretes Beispiel.",
         "Untersuche für {topic} zwei Abläufe und begründe, welcher leichter zu prüfen ist.",
         "Erstelle für {topic} einen Übergabeplan mit Prüfpunkten und Verantwortlichkeiten.",
         "Beschreibe für {topic}, wie man Fehler erfasst und die Behebung überprüft."),
    ),
    (
        ("una biblioteca de barrio que presta tabletas", "un observatorio con registros de nubosidad",
         "una huerta comunitaria con riego compartido", "un archivo de grabaciones de radio local",
         "una escuela que inventaría su material deportivo", "un centro cultural que rota sus exposiciones"),
        ("Redacta una guía práctica para {topic} e incluye un ejemplo de principio a fin.",
         "Compara dos formas de organizar {topic} y explica cómo medir el resultado.",
         "Prepara una sesión de formación para {topic} con ejercicios y soluciones.",
         "Diseña un plan de continuidad para {topic} durante una interrupción de tres días."),
    ),
)


def normalize(prompt):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", prompt).casefold()))


def excluded_prompts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from excluded_prompts(item)
    elif isinstance(value, dict):
        fields = ("prompt", "prompts", "text", "question", "instruction", "input",
                  "items", "cases", "records", "examples", "data")
        for field in fields:
            if field in value:
                yield from excluded_prompts(value[field])
        messages = value.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") != "tool":
                    yield from excluded_prompts(message.get("content"))


def read_prompt_file(path):
    with path.open(encoding="utf-8") as source:
        try:
            yield json.load(source)
            return
        except json.JSONDecodeError:
            source.seek(0)
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc


def load_exclusions(paths):
    unique = set()
    for path in paths:
        found = {normalized for value in read_prompt_file(path) for prompt in excluded_prompts(value)
                 if (normalized := normalize(prompt))}
        if not found:
            raise ValueError(f"No readable evaluation prompts in {path}")
        unique.update(found)
    return sorted(unique)


def external_prompts(value):
    if isinstance(value, str):
        yield "external", value
    elif isinstance(value, list):
        for item in value:
            yield from external_prompts(item)
    elif isinstance(value, dict):
        domain = value.get("domain")
        if not isinstance(domain, str) or not domain.strip():
            domain = "external"
        for key in ("prompt", "instruction"):
            prompt = value.get(key)
            if isinstance(prompt, str) and prompt.strip():
                yield domain, prompt
                return
        messages = value.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    content = "\n".join(part["text"] for part in content
                                        if isinstance(part, dict) and part.get("type") == "text"
                                        and isinstance(part.get("text"), str))
                if isinstance(content, str):
                    yield domain, content


def load_external_prompts(paths):
    for path in paths:
        for value in read_prompt_file(path):
            yield from external_prompts(value)


def supports_fuzzy_match(prompt):
    return len(prompt) >= 16 and len(prompt.split()) >= 4


def build_bank(exclusions, threshold, rng, *, external=(), include_builtin=True, seen=()):
    exact = set(exclusions)
    fuzzy = [(f" {prompt} ", set(prompt.split())) for prompt in exact if supports_fuzzy_match(prompt)]
    seen = set(seen)

    def screen(prompts):
        for name, prompt in prompts:
            normalized = normalize(prompt)
            if not normalized or normalized in seen or normalized in exact:
                continue
            if supports_fuzzy_match(normalized):
                padded = f" {normalized} "
                words = set(normalized.split())
                if any(padded in other or other in padded
                       or len(words & tokens) / len(words | tokens) >= threshold
                       for other, tokens in fuzzy):
                    continue
            seen.add(normalized)
            yield name, prompt

    bank = list(screen(external))
    if not include_builtin:
        return bank
    domains = {name: [template.format(topic=topic) for topic in topics for template in templates]
               for name, (topics, templates) in DOMAINS.items()}
    domains["multilingual"] = [template.format(topic=topic) for topics, templates in MULTILINGUAL
                              for topic in topics for template in templates]
    kept = []
    for name, prompts in domains.items():
        rng.shuffle(prompts)
        kept.append(list(screen((name, prompt) for prompt in prompts)))
    bank.extend(domain[index] for index in range(max(map(len, kept))) for domain in kept if index < len(domain))
    return bank


def load_collected(path):
    seen = set()
    written = 0
    needs_newline = False
    with path.open(encoding="utf-8", newline="") as source:
        for number, line in enumerate(source, 1):
            needs_newline = not line.endswith("\n")
            if not line.strip():
                continue
            try:
                prompt = json.loads(line)["prompt"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid rollout at {path}:{number}") from exc
            if not isinstance(prompt, str) or not (normalized := normalize(prompt)):
                raise ValueError(f"Invalid rollout prompt at {path}:{number}")
            seen.add(normalized)
            written += 1
    return written, seen, needs_newline


def request_rollout(plan, args):
    domain, prompt, temperature, effort, seed = plan
    thinking = effort != "off"
    # ingine-serve uses "none" for zero effort.
    reasoning_effort = effort if thinking else "none"
    payload = {
        "model": args.model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "enable_thinking": thinking, "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
        "top_k": 0, "top_p": 1.0, "seed": seed,
    }
    request = Request(args.url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    for attempt in range(args.retries + 1):
        try:
            with urlopen(request, timeout=args.timeout) as response:
                choice = json.load(response)["choices"][0]
            message = choice["message"]
            content, reasoning = message.get("content") or "", message.get("reasoning_content") or ""
            reason = choice.get("finish_reason")
            if (not isinstance(content, str) or not isinstance(reasoning, str)
                    or (not thinking and reasoning) or (not content and not reasoning) or reason == "tool_calls"):
                raise ValueError("completion cannot be represented as a text rollout")
            return {"prompt": prompt, "response": content, "reasoning": reasoning, "thinking": thinking,
                    "reasoning_effort": reasoning_effort,
                    "temperature": temperature, "finish_reason": reason or "", "domain": domain}, None
        except HTTPError as exc:
            error = f"HTTP {exc.code}"
            if exc.code not in (408, 429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError) as exc:
            error = str(exc)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"invalid completion: {exc}"
            break
        if attempt < args.retries:
            time.sleep(2 ** attempt)
    return None, error


def parse_mix(value, kind):
    mix = {}
    try:
        for item in value.split(","):
            label, weight = item.split(":")
            label = label.strip()
            if kind == "temperature":
                label = float(label)
                if not 0 <= label <= 2:
                    raise ValueError("temperatures must be in [0, 2]")
            elif label not in ("off", "low", "high"):
                raise ValueError("efforts must be off, low or high")
            weight = float(weight)
            if label in mix or not math.isfinite(weight) or weight < 0:
                raise ValueError("labels must be unique and weights finite and nonnegative")
            mix[label] = weight
        total = sum(mix.values())
        if not 0 < total < math.inf:
            raise ValueError("total weight must be finite and positive")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated {kind}:weight pairs ({exc})") from exc
    return tuple(mix.items())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-files", type=Path, nargs="+", action="extend", required=True,
                        help="Evaluation JSON or JSONL files containing prompts")
    parser.add_argument("--prompt-files", type=Path, nargs="+", action="extend", default=[],
                        help="Collect from these JSON or JSONL prompt files before the built-in bank")
    parser.add_argument("--no-builtin-bank", action="store_true", help="Collect only from external prompts")
    parser.add_argument("--output", "--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Append missing rollouts up to the total count")
    parser.add_argument("--url", default="http://127.0.0.1:8834/v1")
    parser.add_argument("--model", default="inkling-2-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--count", type=int, default=432)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--jaccard", type=float, default=0.72, help="Reject overlap at or above this cutoff")
    parser.add_argument("--temp-mix", type=lambda value: parse_mix(value, "temperature"),
                        default="0.0:0.25,0.8:0.50,1.0:0.25", help="Comma-separated temperature:weight pairs")
    parser.add_argument("--effort-mix", type=lambda value: parse_mix(value, "effort"),
                        default="off:0.20,low:0.40,high:0.40", help="Comma-separated effort:weight pairs")
    args = parser.parse_args(argv)
    if (min(args.count, args.concurrency, args.max_tokens) < 1 or args.retries < 0
            or not 0 < args.timeout < float("inf") or not 0 < args.jaccard <= 1):
        parser.error("Invalid count, concurrency, budget, retry, timeout or Jaccard cutoff")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    rng = random.Random(args.seed)
    append = args.resume and args.output.exists()
    written, seen, needs_newline = 0, set(), False
    try:
        output = args.output.resolve()
        if any(output == path.resolve() for path in (*args.exclude_files, *args.prompt_files)):
            raise ValueError("Output path must differ from each exclusion or prompt file")
        if append:
            written, seen, needs_newline = load_collected(args.output)
        if written >= args.count:
            print(f"Already collected {written} rollouts in {args.output}", flush=True)
            return 0
        exclusions = load_exclusions(args.exclude_files)
        bank = build_bank(exclusions, args.jaccard, rng, external=load_external_prompts(args.prompt_files),
                          include_builtin=not args.no_builtin_bank, seen=seen)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    remaining = args.count - written
    if len(bank) < remaining:
        parser.error(f"Only {len(bank)} disjoint prompts available for {remaining} remaining rollouts")
    temps, temp_weights = zip(*args.temp_mix)
    efforts, effort_weights = zip(*args.effort_mix)
    plans = ((domain, prompt, rng.choices(temps, temp_weights)[0],
              rng.choices(efforts, effort_weights)[0], rng.randrange(2 ** 63))
             for domain, prompt in bank)
    print(f"Collecting {remaining} rollouts from {len(bank)} prompts screened against "
          f"{len(exclusions)} evaluation prompts", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if append else "w", encoding="utf-8") as out, ThreadPoolExecutor(args.concurrency) as pool:
        pending = deque()
        index = 0
        while written < args.count:
            while len(pending) < min(args.concurrency, args.count - written):
                plan = next(plans, None)
                if plan is None:
                    break
                pending.append(pool.submit(request_rollout, plan, args))
            if not pending:
                break
            row, error = pending.popleft().result()
            index += 1
            if error is not None:
                print(f"rollout {index}: {error}", file=sys.stderr, flush=True)
                continue
            if needs_newline:
                out.write("\n")
                needs_newline = False
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
    print(f"Wrote {written}/{args.count} rollouts to {args.output}", flush=True)
    return int(written != args.count)


if __name__ == "__main__":
    raise SystemExit(main())
