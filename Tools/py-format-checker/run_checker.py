#!/usr/bin/env python3
"""
run_checker.py - run the PyFormatChecker against every C file recorded in
                  compile_commands.json and collect the output.

Usage:
    cd /path/to/cpython
    python3 Tools/py-format-checker/run_checker.py [output.txt] [--jobs N]

By default only call site(s) with at least one mismatch are printed
(PY_FMT_ERROR_ONLY=1).  Pass --verbose to print all call site(s).

The script re-runs each compilation command with:
  - the compiler replaced by clang-21 (override with --clang)
  - --target=<triple> injected when --target is given (e.g. i686-linux-gnu
    to check with 32-bit long/size_t; no cross-compilation of the plugin
    itself is required — only the target type-system changes)
  - -fplugin=<plugin.so> appended
  - -w to suppress normal warnings (we only want plugin output)
  - -fsyntax-only appended (unless -c/-E/-S is already present)
  - output redirected to /dev/null; real object-file output suppressed

Results are written to Tools/py-format-checker/reports/py_format_report[_<target>].txt
(default) or the path you supply as the first argument.  When --target is given the
target triple is embedded in the default filename, e.g.
  reports/py_format_report_i686-linux-gnu.txt

Environment variables:

- `PY_FMT_ERROR_ONLY`: Default `1`. Set to `0` to print every checked call site,
    not just those with errors (equivalent to --verbose).

- `PY_FMT_INTEGRAL_CHECK_MODE`: Controls integer width/sign checking. Values:
    - `off` — accept any integer type for integer specifiers; no
      width or signedness check.
    - `standard` (default) — bit-width must match the specifier (C99-style); signedness
      is ignored (useful when the codebase freely mixes int/unsigned).
    - `full` — both bit-width and signedness must match.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PLUGIN_SO = Path(__file__).parent / "build" / "PyFormatChecker.so"
REPORTS_DIR = Path(__file__).parent / "reports"
CLANG = "clang-21"
PREFIX = "[py-fmt]"


def reformat_output(raw_text: str) -> str:
    """Parse and reformat raw [py-fmt] plugin output into a human-readable report.

    The plugin emits structured lines prefixed with '[py-fmt]'
    (func:/loc:/fmt:/arg[N]/...).  This function:

    1. Strips the '[py-fmt]' sentinel prefix and splits lines into blocks,
       each starting with a 'func:' line.
    2. Normalises 'loc:' paths by collapsing '../' segments and stripping
       a leading './'.
    3. Deduplicates blocks that share the same 'loc:' (keeps first occurrence).
    4. Sorts blocks lexicographically by (file, line-number).
    5. Filters out intentional UNKNOWN_SPEC blocks from exception-listed test
       files (see is_ignorable_test_unknown).
    6. Prepends an 'Error N/M:' or 'Call site N/M:' header to each block and
       joins blocks with a '---' rule.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in raw_text.splitlines():
        if not line.startswith("[py-fmt]"):
            continue
        # Strip the "[py-fmt]" prefix (8 chars) plus optional single space.
        body = line[8:]
        if body.startswith(" "):
            body = body[1:]
        # A 'func:' line marks the start of a new block.
        if body.startswith("func:"):
            if current:
                blocks.append(current)
            current = [body]
        else:
            if current:
                current.append(body)
    if current:
        blocks.append(current)

    # Normalize `loc:` paths by collapsing any up-level segments (../)
    for block in blocks:
        for i, line in enumerate(block):
            if line.startswith("loc:"):
                loc = line.split(":", 1)[1].strip()
                if ":" in loc:
                    path, _, lineno = loc.rpartition(":")
                    norm = os.path.normpath(path)
                    if norm.startswith("./"):
                        norm = norm[2:]
                    block[i] = f"loc:      {norm}:{lineno}"
                else:
                    norm = os.path.normpath(loc)
                    if norm.startswith("./"):
                        norm = norm[2:]
                    block[i] = f"loc:      {norm}"
                break

    # Deduplicate blocks that share the same `loc:`. Keep the first
    # occurrence for each location and drop later duplicates.
    seen_locs = set()
    unique_blocks = []
    for block in blocks:
        loc_line = next(
            (line for line in block if line.startswith("loc:")), None
        )
        loc_val = loc_line.split(":", 1)[1].strip() if loc_line else ""
        if loc_val in seen_locs:
            continue
        seen_locs.add(loc_val)
        unique_blocks.append(block)
    blocks = unique_blocks

    # Sort blocks by location: extract "loc: file:line" and sort
    # lexicographically on (file, line-number).
    def block_sort_key(block: list[str]) -> tuple[str, int]:
        for line in block:
            if line.startswith("loc:"):
                loc = line.split(":", 1)[1].strip()  # "file.c:123"
                if ":" in loc:
                    path, _, lineno = loc.rpartition(":")
                    try:
                        return (path, int(lineno))
                    except ValueError:
                        return (loc, 0)
                return (loc, 0)
        return ("", 0)

    blocks.sort(key=block_sort_key)

    # Ignore UNKNOWN_SPEC blocks from explicitly listed test files: these are
    # intentional tests for unrecognised specs (e.g. the "// Unrecognized"
    # section in _testlimitedcapi/unicode.c).  A block qualifies for
    # suppression when:
    #   - the block contains an UNKNOWN_SPEC diagnostic, AND
    #   - loc: path is a key in the exception_lines dict, AND
    #   - the source line at the reported location matches one of the
    #     expected strings for that file.
    def is_ignorable_test_unknown(block):
        if not any("UNKNOWN_SPEC" in line for line in block):
            return False

        loc_line = next(
            (line for line in block if line.startswith("loc:")), None
        )
        if loc_line is None:
            return False

        loc = loc_line.split(":", 1)[1].strip()
        if ":" not in loc:
            return False

        path, _, lineno_str = loc.rpartition(":")

        try:
            lineno = int(lineno_str)
        except ValueError:
            return False

        exception_lines = {
            "Modules/_testlimitedcapi/unicode.c": [
                'CHECK_FORMAT_2("%u %? %u", NULL, 1, 2);',
            ]
        }

        if path not in exception_lines:
            return False

        # Try to find the source file relative to the repo root
        src_path = Path(path)
        if not src_path.is_absolute():
            # The loc path is workspace-relative; resolve from repo root
            src_path = Path(__file__).parent.parent.parent / path
        if not src_path.exists():
            return False

        try:
            src_lines = src_path.read_text(errors="replace").splitlines()
        except OSError:
            return False

        if lineno < 1 or lineno > len(src_lines):
            return False

        src_line = src_lines[lineno - 1].strip()

        return src_line in exception_lines[path]

    blocks = [b for b in blocks if not is_ignorable_test_unknown(b)]

    total = len(blocks)
    formatted: list[str] = []
    for n, block in enumerate(blocks, 1):
        has_mismatch = any(
            "MISMATCH" in line or "MISSING_ARG" in line or "SURPLUS" in line
            for line in block
        )
        label = "Error" if has_mismatch else "Call site"
        formatted.append(f"{label} {n}/{total}:\n\n" + "\n".join(block))

    if not formatted:
        return ""
    return "\n\n---\n\n".join(formatted) + "\n"


def process_entry(
    entry: dict,
    plugin: str,
    clang: str,
    extra_env: dict,
    target: str | None = None,
) -> str:
    """Run one compile_commands entry; return plugin stdout lines (or '')."""
    cmd: list[str] = entry.get("arguments") or entry["command"].split()
    directory = entry["directory"]

    # Replace compiler with the requested clang binary
    cmd[0] = clang

    # Drop object-file output to avoid clobbering build artefacts
    new_cmd = []
    skip_next = False
    for tok in cmd:
        if skip_next:
            skip_next = False
            new_cmd.append("/dev/null")
            continue
        if tok == "-o":
            skip_next = True
            new_cmd.append(tok)
            continue
        if tok.startswith("-o"):
            new_cmd.append("-o/dev/null")
            continue
        new_cmd.append(tok)

    # Inject target triple before other extra flags so it takes effect first.
    if target:
        new_cmd += [f"--target={target}"]

    new_cmd += [
        f"-fplugin={plugin}",
        "-w",  # silence regular warnings – only want plugin output
    ]

    # Only add -fsyntax-only when no conflicting action flags are present.
    # This avoids combining it with flags like -c, -E, or -S, and prevents
    # adding a redundant -fsyntax-only if it is already in the command.
    conflicting_flags = {"-c", "-E", "-S", "-fsyntax-only"}
    if not any(tok.split("=", 1)[0] in conflicting_flags for tok in new_cmd):
        new_cmd.append("-fsyntax-only")
    new_cmd = [t for t in new_cmd if t]

    try:
        env = os.environ.copy()
        env.update(extra_env)
        result = subprocess.run(
            new_cmd,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(PREFIX)
        ]
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return f"# TIMEOUT: {entry['file']}"
    except Exception as exc:
        return f"# ERROR({exc}): {entry['file']}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("output", nargs="?", default=None)
    ap.add_argument(
        "--jobs", "-j", type=int, default=max(1, os.cpu_count() or 1)
    )
    ap.add_argument(
        "--plugin", default=str(PLUGIN_SO), help="Path to PyFormatChecker.so"
    )
    ap.add_argument(
        "--db",
        default="compile_commands.json",
        help="Path to compile_commands.json",
    )
    ap.add_argument(
        "--clang",
        default=CLANG,
        help="clang binary to use (default: clang-21)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print all call site(s), not just those with errors "
        "(sets PY_FMT_ERROR_ONLY=0)",
    )
    ap.add_argument(
        "--target",
        default=None,
        metavar="TRIPLE",
        help="Clang target triple to pass as --target=TRIPLE (e.g. "
        "i686-linux-gnu to check with 32-bit long/size_t). "
        "The plugin binary itself is not affected.",
    )
    args = ap.parse_args()

    extra_env: dict[str, str] = {}
    if args.verbose:
        extra_env["PY_FMT_ERROR_ONLY"] = "0"

    # Resolve the target triple: use the explicit --target value if given,
    # otherwise query the clang binary for its default host triple so the
    # report filename always reflects which ABI was checked.
    target = args.target
    if not target:
        try:
            target = subprocess.check_output(
                [args.clang, "--print-target-triple"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass  # leave target as None; filename will be generic

    plugin = Path(args.plugin).resolve()
    if not plugin.exists():
        sys.exit(
            f"Plugin not found: {plugin}\n"
            "Build it first:\n"
            "  cd Tools/py-format-checker && mkdir -p build && cd build\n"
            "  cmake .. && make -j$(nproc)"
        )

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"compile_commands.json not found at {db_path}")

    entries = json.loads(db_path.read_text())
    print(
        f"Processing {len(entries)} TUs with {args.jobs} workers ...",
        flush=True,
    )

    seen: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(
                process_entry,
                e,
                str(plugin),
                args.clang,
                extra_env,
                target,
            ): e
            for e in entries
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 50 == 0 or done == len(entries):
                print(f"  {done}/{len(entries)}", flush=True)
            out = fut.result()
            if out:
                seen.append(out)

    result_text = "\n".join(seen) + "\n"
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_{target}" if target else ""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"py_format_report{suffix}.txt"
    report_text = reformat_output(result_text)
    output_path.write_text(report_text)

    # Summary (count from reformatted output to reflect actual reported issues)
    report_lines = report_text.splitlines()
    call_sites = sum(
        1
        for line in report_lines
        if line.startswith("Error ") or line.startswith("Call site ")
    )
    mismatches = sum(
        1
        for line in report_lines
        if " MISMATCH" in line
        or " MISSING_ARG" in line
        or " SURPLUS" in line
        or " UNKNOWN_SPEC" in line
    )
    hints = sum(1 for line in report_lines if "hint:" in line)
    mode = "all call site(s)" if args.verbose else "error only"
    print(f"\nDone ({mode}).")
    print(f"  {call_sites} call site(s) with issues reported.")
    print(f"  {mismatches} mismatch/missing/surplus argument(s).")
    print(report_text)
    if hints:
        print(
            f"  {hints} auto-correctable format string(s) (see 'hint:' lines)."
        )
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
