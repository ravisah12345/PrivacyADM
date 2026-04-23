# COM6020M — Algorithm Transparency in ADM Systems
## Privacy and Data Protection | York St John University

This repository contains the practical appendix for the 
COM6020M report on Privacy Challenges in Automated 
Decision-Making Systems.

---

## What this demonstrates

| Code Section | Report Section |
|---|---|
| Decision Tree model | Section 2.1 — Black Box Problem |
| 6 personal attributes | Section 2.2 — Data Collection |
| Credit score risk flag | Section 2.3 — Inferential Profiling |
| Per-decision explanation | Section 3.1 — XAI / GDPR Article 22 |
| Privacy risk summary | Recommendation 1 — Algorithmic Auditing |

---

## Files

- `algorithm_transparency.py` — Full Python script
- `algorithm_transparency.png` — Output chart

---

## How to run

Install dependencies:
pip install scikit-learn matplotlib pandas numpy

Run the script:
python algorithm_transparency.py

---

## Output

The script produces:
- Feature importance analysis
- Per-decision explanation for two applicants
- Privacy risk assessment summary
- Chart saved as algorithm_transparency.png
