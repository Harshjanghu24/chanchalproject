# DATA_NOTES.md — CDAS Stage 1 (audit only)

## Source files in workspace (D:\chanchal project)

| File | Role | Rows | Columns |
|------|------|------|---------|
| AI_Literacy_questionnaire_data_.csv | MAIN DATASET | 1,205 | 58 |
| AI_Literacy_survey.pdf | Survey instrument (question text), not a data file | — | — |

The CSV is the only analyzable dataset. The PDF is the Dai et al. (2020) AI Literacy survey instrument (question wording only).

## Main dataset structure

- **Rows:** 1,205
- **Columns:** 58
- **Missingness:** 0 missing values in every column (no NaN/blank cells anywhere).
- **Dtypes:** 5 numeric ID/demographic columns (user_id, year, Grade, Age, AI-course hours) + 3 string (Gender, Country, Study programme) + 50 Likert item columns (all int64, values on a 1–5 scale unless coded 0 = N/A).

### Column groups (from CSV header, verified against PDF question text)

| Group | Items | Meaning (per PDF) |
|-------|-------|-------------------|
| Demographics | user_id, year, Grade, Age, Gender, Country, Study programme, AI-course hours | Background |
| L (AI Literacy) | L1–L5 | Knowledge of AI capabilities (image/speech recognition, translation, voice assistant, prediction) |
| RE (AI readiness) | RE1–RE6 | AI in daily life, convenience, personalization, stimulating thinking, following instructions |
| R (Relevance) | R1–R6 | AI changing the world, usefulness, basics, future connection, interest, everyday connection |
| SG (Social Goods) | SG1–SG5 | Using AI to serve/others/help humanity |
| CM (Career motivation) | CM1–CM4 | Future help, good job, interesting work, future interests |
| **A (AI anxiety)** | **A1–A5** | Worry about AI trouble, unsure future, future fail because of AI, uneasy, pressured by AI advancement |
| IM (Intrinsic motivation) | IM1–IM4 | Curiosity, challenge, thorough understanding, liking course content |
| S (Satisfaction) | S1–S5 | Satisfied, good, interesting, rewarded |
| C (Confidence) | C1–C5 | Good grades, learn well if work hard, difficult content, basic concepts, complex content |
| BI (Behavioural intention) | BI1–BI5 | Continue learning, pay attention, stay tuned, use AI tools to learn/solve problems |

Source citation (PDF): Dai, Y., Chai, C.-S., Lin, P.-Y., Jong, M. S.-Y., Guo, Y., & Qin, J. (2020). Promoting Students’ Well-Being by Developing Their Readiness for the Artificial Intelligence Age. Sustainability, 12(16), 6597.

## Construct inventory: present vs missing

| Construct | Status | Evidence |
|-----------|--------|----------|
| AI anxiety | **EXISTS** | Columns A1–A5 (5 items, values 1–5, no zeros, no missing) |
| AI literacy | **EXISTS** | Columns L1–L5 (5 items, values 1–5, no zeros) |
| Autonomy/control | **MISSING** | No column named autonomy/control. Closest proxy: **RE6** "I am confident that AI technology will do things following my instructions" (perceived control over AI). |
| AI usage | **MISSING** | No column measuring actual AI usage frequency/intensity. Closest proxy: **"How many hours of AI-related courses have you taken"** (course exposure, 742 of 1,205 rows = 0 hours; not a usage measure). |
| Perceived benefit | **MISSING** | No column named perceived benefit. Closest proxy: **CM1** "I think learning AI is helpful to my future" and **R2** "Learning AI related knowledge is very useful for me". |
| Perceived risk | **MISSING** | No column named perceived risk exists, and no proxy is used. A3 belongs to the AI anxiety target (A1–A5), so using it as a risk proxy would be target leakage and is explicitly disallowed. |
| Self-doubt / imposter | **MISSING** | No column named self-doubt or imposter. Closest proxy: **C (Confidence) items C1–C5**, read in inverse (low confidence ≈ self-doubt); e.g. C3 "I believe I can understand the most difficult content in AI classes", C5 complex content. |

**Rule followed:** no columns or values were invented. Every "missing" construct is flagged with the closest real proxy explicitly labelled as a proxy.

## Figshare AI Literacy Questionnaire — target construction

- **Target:** mean of A1–A5 (AI anxiety), computed per row over the 5 anxiety items.
- A1–A5 value range observed: 1 to 5 (no 0-coded cells on these items), so no N/A recoding needed for the target itself.

### Cronbach's alpha for A1–A5

- **Alpha = 0.8820** (n = 1,205, all rows complete on A1–A5)
- Computed as Cronbach's alpha over the 5 items: k/(k−1) × (1 − Σ item variances / total score variance).

### N/A coded as 0 (treat 0 as NaN) — zero cell counts per scale

| Scale | Items | Zero-coded cells (0 treated as N/A) |
|-------|-------|-------------------------------------|
| IM (Intrinsic motivation) | IM1–IM4 | 1,398 |
| C (Confidence) | C1–C5 | 1,799 |
| S (Satisfaction) | S1–S5 | 1,990 |
| BI (Behavioural intention) | BI1–BI5 | 956 |

Per-item zero counts:

- IM: IM1=257, IM2=320, IM3=328, IM4=493
- C: C1=386, C2=332, C3=348, C4=311, C5=422
- S: S1=315, S2=513, S3=246, S4=516, S5=400
- BI: BI1=324, BI2=178, BI3=156, BI4=154, BI5=144

These zeros are present in the raw data and should be recoded to NaN before any scale scoring; they are NOT missingness flags (isna reports 0 everywhere) — they are explicit N/A codes.

### Repeated user_id

- Total rows: 1,205
- Unique user_id values: 1,146
- Rows with a duplicated user_id: 118 (59 user_ids each appear exactly twice; no user_id appears more than twice)
- Duplicated ids: 7, 10, 11, 12, 13, 27, 29, 31, 39, 41, 47, 49, 56, 58, 59, 63, 64, 78, 84, 92, 93, 94, 96, 100, 103, 114, 117, 118, 124, 128, 140, 150, 152, 155, 157, 159, 172, 174, 187, 197, 199, 201, 206, 207, 209, 210, 217, 221, 222, 224, 228, 243, 248, 249, 256, 257, 260, 264, 265

## Files produced this stage

- `audit_datasets.py` — created (audit script; starter file did not exist)
- `audit_report.txt` — full column/dtype/missingness/zero-count dump for the CSV
- `DATA_NOTES.md` — this file

## Status

Stage 1 complete. Awaiting go-ahead for Stage 2 (modeling). No modeling has been run.
