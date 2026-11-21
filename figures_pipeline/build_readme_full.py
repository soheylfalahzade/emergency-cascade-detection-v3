"""Build the root README.md = short header + the complete report.

Reads docs/report/REPORT.md (produced by build_report.py) and the header of the
current README.md, and writes README.md so that it contains the full report with
repository-relative paths. Idempotent: the header is everything before the first
of the markers below.
"""
import re
import sys

REPORT = "docs/report/REPORT.md"
README = "README.md"
MARKERS = ("## Abstract", "> **Full report:**", "## Contents")

rep = open(REPORT, encoding="utf-8").read()
cur = open(README, encoding="utf-8").read()

# header of the current README
cuts = [cur.find(m) for m in MARKERS if cur.find(m) >= 0]
if not cuts:
    sys.exit("NO_HEADER_MARKER")
head = cur[:min(cuts)].rstrip() + "\n\n"
head = head.replace("Section 3.5 defines them", "Section 3.6 defines them")

# body of the report: from "## Abstract", without its own "How to read" callout
k = rep.find("## Abstract")
if k < 0:
    sys.exit("NO_ABSTRACT_IN_REPORT")
body = rep[k:]
body = re.sub(r"> \[!IMPORTANT\]\n(?:>.*\n)+\n?(?:---\n)?", "", body, count=1)
body = re.sub(r"\n---\n\s*\n---\n", "\n---\n", body)

# paths: report lives in docs/report/, README in the repository root
body = body.replace("../../phase2_bridge/", "phase2_bridge/")
body = body.replace("../../docs/", "docs/")
body = body.replace("../readme_figures/", "docs/readme_figures/")
body = body.replace("../paper_figures/", "docs/paper_figures/")
body = body.replace("docs/report/REPORT.md", "README.md")

body = body.replace(
    "python figures_pipeline/build_report.py        # this report from result files",
    "python figures_pipeline/build_report.py        # docs/report/REPORT.md from result files\n"
    "python figures_pipeline/build_readme_full.py   # this README = header + full report (run last)")

left = re.findall(r'(?:src|href)="(\.\./[^"]*)"|\]\((\.\./[^)]*)\)', body)
if left:
    print("UNRESOLVED_RELATIVE_PATHS", left[:5]); sys.exit(1)

out = head + body
if "\r" in out:
    out = out.replace("\r\n", "\n")
open(README, "w", encoding="utf-8", newline="\n").write(out)
print("README rebuilt:", len(out.splitlines()), "lines")
