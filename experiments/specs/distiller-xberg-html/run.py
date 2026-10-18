#!/usr/bin/env python3
"""Bounded local CLI comparison; raw results stay in experiments/runs."""
import argparse
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin

SPEC = Path(__file__).resolve().parent
ROOT = SPEC.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Source(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.code = []
        self.math = []
        self.links = []
        self.pre_depth = 0
        self.math_depth = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.pre_depth += 1
            self.code.append("")
        if tag == "br" and self.pre_depth:
            self.code[-1] += "\n"
        if tag == "math":
            if not self.math_depth:
                self.math.append("")
            self.math_depth += 1
        if self.math_depth:
            self.math[-1] += repr((tag, sorted(attrs)))
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href and not href.startswith("#"):
                self.links.append(href)

    def handle_endtag(self, tag):
        if self.math_depth:
            self.math[-1] += "</" + tag + ">"
        if tag == "math":
            self.math_depth -= 1
        if tag == "pre":
            self.pre_depth -= 1

    def handle_data(self, data):
        if self.pre_depth:
            self.code[-1] += data
        if self.math_depth:
            self.math[-1] += "".join(data.split())


def code_blocks(markdown):
    blocks, fence, content = [], None, []
    for line in markdown.splitlines(keepends=True):
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\n"))
        if fence is None and match:
            fence = match[1]
            content = []
        elif fence and match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
            blocks.append("".join(content))
            fence = None
        elif fence:
            content.append(line)
    return blocks


def evaluate(html, markdown, url):
    source = Source(html)
    blocks = code_blocks(markdown)
    output_math = Source(markdown).math
    links = {urljoin(url, target) for target in source.links}
    output_links = {urljoin(url, target) for target in Source(markdown).links}
    output_links.update(urljoin(url, target) for target in re.findall(r"\]\(([^)\s]+)\)", markdown))
    return {
        "code_exact": sum(code in blocks or (not code.endswith("\n") and code + "\n" in blocks) for code in source.code),
        "code_total": len(source.code),
        "math_structure_retained": sum(expr in output_math for expr in source.math),
        "math_total": len(source.math),
        "links_retained": len(links & output_links),
        "links_total": len(links),
        "empty_table_cell": bool(re.search(r"\|\s*1\s*\|\s*\|\s*\[?3", markdown) or re.search(r"<td>\s*</td>", markdown)),
        "table_spans": 'colspan="2"' in markdown and 'rowspan="2"' in markdown,
    }


def self_check():
    html = '<pre> a\n\n2\n</pre><math><msup><mi>x</mi><mn>2</mn></msup></math>'
    good = '````rust\n a\n\n2\n````\n<math><msup><mi>x</mi><mn>2</mn></msup></math>'
    assert evaluate(html, good, "http://example/")["code_exact"] == 1
    assert evaluate(html, good.replace(" a", "a"), "http://example/")["code_exact"] == 0
    assert evaluate(html, good, "http://example/")["math_structure_retained"] == 1
    assert evaluate(html, good.replace("<msup>", "<mrow>"), "http://example/")["math_structure_retained"] == 0
    assert evaluate('<math><mspace width="2em"></mspace></math>', '<math><mspace></mspace></math>', "http://example/")["math_structure_retained"] == 0
    assert evaluate('<a href="/a">a</a>', 'http://example/a', "http://example/")["links_retained"] == 0
    assert code_blocks('```\na\n```\n') == ["a\n"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("Evaluator detects indentation loss and math-tree changes.")
        return
    subprocess.run(["just", "experiment-ready", SPEC.name], cwd=ROOT, check=True)
    manifest = json.loads((SPEC / "manifest.json").read_text())
    run = ROOT / manifest["evidence"]["raw_runs"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, exist_ok=False)
    records = []
    with tempfile.TemporaryDirectory(prefix="distiller-xberg-") as temporary:
        work = Path(temporary)
        shutil.copytree(SPEC / "fixtures", work / "fixtures")
        (work / "home").mkdir()
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work / "home"), "TMPDIR": str(work)}
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(work / "fixtures"))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        identities = {v["name"]: {"path": v["command"][0], "sha256": digest(Path(v["command"][0]))} for v in manifest["variants"]}
        (run / "identity.json").write_text(json.dumps({"manifest_sha256": digest(SPEC / "manifest.json"), "source_digest": manifest["source"]["content_digest"], "runner_sha256": digest(Path(__file__)), "tools": identities, "environment_names": list(env), "temporary_directory": temporary}, indent=2))
        try:
            with (run / "terminal.jsonl").open("a") as events:
                for fixture in ("article.html", "technical.html"):
                    url = f"http://127.0.0.1:{server.server_port}/{fixture}"
                    for repetition in range(manifest["repetitions"] + 1):
                        variants = manifest["variants"] if repetition % 2 == 0 else list(reversed(manifest["variants"]))
                        for variant in variants:
                            label = f'{fixture[:-5]}-{variant["name"]}-{repetition}'
                            command = [arg.replace("{url}", url).replace("{fixture}", str(work / "fixtures" / fixture)) for arg in variant["command"]]
                            started = datetime.now(timezone.utc).isoformat()
                            before = time.perf_counter()
                            with (run / f"{label}.md").open("wb") as stdout, (run / f"{label}.stderr").open("wb") as stderr:
                                process = subprocess.Popen(["/usr/bin/time", "-l", *command], cwd=work, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                                try:
                                    status = process.wait(timeout=manifest["environment"]["resources"]["timeout_seconds"])
                                except BaseException as failure:
                                    os.killpg(process.pid, signal.SIGKILL)
                                    process.wait()
                                    if not isinstance(failure, subprocess.TimeoutExpired):
                                        raise
                                    status = "timeout"
                            elapsed = time.perf_counter() - before
                            output = run / f"{label}.md"
                            error = run / f"{label}.stderr"
                            assert output.stat().st_size + error.stat().st_size <= manifest["environment"]["resources"]["max_output_bytes"], "output budget exceeded; retained for diagnosis"
                            rss = re.search(r"(\d+)\s+maximum resident set size", error.read_text())
                            record = {"fixture": fixture, "variant": variant["name"], "repetition": repetition, "warmup": repetition == 0, "command": command, "started": started, "ended": datetime.now(timezone.utc).isoformat(), "exit": status, "wall_seconds": elapsed, "peak_rss_bytes": int(rss[1]) if rss else None, "output": output.name, "output_sha256": digest(output), "stderr_sha256": digest(error)}
                            if status == 0:
                                record["fidelity"] = evaluate((work / "fixtures" / fixture).read_text(), output.read_text(), url)
                            events.write(json.dumps(record) + "\n")
                            events.flush()
                            records.append(record)
                            assert status == 0, f"{label}: exit {status}, inspect {error}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    summary = {}
    for fixture in ("article.html", "technical.html"):
        summary[fixture] = {}
        for variant in manifest["variants"]:
            rows = [r for r in records if r["fixture"] == fixture and r["variant"] == variant["name"] and not r["warmup"]]
            assert len(rows) == manifest["repetitions"]
            assert all(row["fidelity"] == rows[0]["fidelity"] for row in rows), "non-deterministic fidelity"
            times = [row["wall_seconds"] for row in rows]
            summary[fixture][variant["name"]] = {"fidelity": rows[0]["fidelity"], "median_ms": statistics.median(times) * 1000, "range_ms": [min(times) * 1000, max(times) * 1000], "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows if row["peak_rss_bytes"] is not None)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
