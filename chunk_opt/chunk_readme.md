1. Experimental Setup

We evaluate four experiment settings:

Exp ID	chunk_size	overlap
A_500_150	500	150
B_800_200	800	200
Base	1000	200
C_1200_250	1200	250
Why these 4 settings?

These represent the four most meaningful regions in chunking space:

A: smaller chunk → high precision

B: medium chunk → balanced

Base: our default

C: large chunk → more context but risky

Procedure

For each configuration, we:

Update chunk_size & overlap in config.yaml.

Re-ingest all documents (ingest_documents.py --reset).

Run 10 evaluation questions automatically via run_phase1_chunking.py.

Collect raw LLM answers in JSONL format.

Apply human scoring (0–1 scale per dimension; total 0–4).

2. Scoring Metrics (0–1 per dimension)
Metric	Meaning
Relevance	Whether the answer matches the question
Correctness	Whether the answer is factually accurate
Completeness	Whether the answer covers essential points
Conciseness	Whether the answer is clear and focused
Total	Sum of four scores (0–4)

This ensures fairness across configurations.

3. Final Results (Average Scores)
Exp ID	chunk_size	overlap	Avg Total (0–4)	Rank
A_500_150	500	150	2.86	1
B_800_200	800	200	2.58	2
Base	1000	200	2.41	3
C_1200_250	1200	250	2.21	4
7. Interpretation of Results
✔ A_500_150 performs best (2.86 / 4)

Small-to-medium chunks with moderate overlap give:

better retrieval precision

fewer “no result found” errors

more complete answers

✔ Medium chunk size (B_800_200) is also strong

It’s similar to values recommended in literature and provides balanced performance.

✔ Default (1000/200) is okay but less stable

Larger chunks tend to blend multiple unrelated topics → lowering overall relevance.

✔ Large chunks (C_1200_250) perform worst

Too much context leads to:

worse retrieval precision

more irrelevant retrievals

incomplete or confused answers

4. Conclusion

The optimal design for this project is:

chunk_size ≈ 500–800
chunk_overlap ≈ 150–200

This gives the best trade-off between context preservation and retrieval precision
and will be adopted in later stages of the system.

5. Files Produced

phase1_chunking_results.jsonl — raw LLM answers

phase1_chunking_report.xlsx — scoring table + line chart

chunking_3d_chart.xlsx — 3D chunking performance visualization