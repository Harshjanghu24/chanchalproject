CDAS: AI Anxiety and Cognitive Dependence

Code for analysing the link between AI anxiety and cognitive dependence on AI in student survey data. It compares Random Forest, Ridge/ElasticNet, XGBoost and a Bayesian network on four public datasets.

Read this first: everything here reports associations, not causes. AI anxiety is treated as a research measure, not a diagnosis, and no clinical cut-offs are used.

Contents
Repository layout
Quick start
Step 1: Set up Python
Step 2: Download the datasets
Step 3: Run the pipeline
Expected outputs
Helper scripts
Design choices
AI-CogSense (AIAS) survey
Limitations
Troubleshooting
Repository layout
chanchalproject/
├── README.md
├── requirements.txt
├── DATA_NOTES.md          notes on the datasets (columns, coding, quirks)
│
├── data_prep.py           builds scale scores, prints sample sizes and correlations
├── train_models.py        Ridge / Random Forest / XGBoost on the Figshare data
├── cv_eval.py             repeated user-grouped cross-validation
├── bayes_net.py           Bayesian network (pgmpy) and graph plot
├── multi_datasets.py      the other three datasets, incl. benefit x cost analysis
│
├── audit_datasets.py      helper: checks the dataset files (see "Helper scripts")
├── inspect_data.py        helper: inspects columns and types
├── peek_values.py         helper: shows the values in each column
├── audit_report.txt       saved output of audit_datasets.py
├── inspect_report.txt     saved output of inspect_data.py
├── peek_report.txt        saved output of peek_values.py
│
├── datasets/              put the downloaded data files here (not redistributed)
├── results/               tables and figures are written here
└── cogsense/reference/    reference material for the AIAS survey
Quick start
bash
git clone https://github.com/Harshjanghu24/chanchalproject.git
cd chanchalproject

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

# download the 4 datasets into datasets/ (see Step 2), then:
python data_prep.py
python train_models.py
python cv_eval.py
python bayes_net.py
python multi_datasets.py

All commands are run from the repository root.

Step 1: Set up Python

Python 3.12 is recommended.

bash
python -m venv .venv

Activate it:

System	Command
Windows (cmd / PowerShell)	.venv\Scripts\activate
macOS / Linux	source .venv/bin/activate

Install the dependencies:

bash
pip install -r requirements.txt

Optional: pip install shap for SHAP plots (only used by the AIAS part, which is not yet included; see below).

Step 2: Download the datasets

The data is not included in this repo, to respect each dataset's licence. Check the terms on each page, then save the files in the datasets/ folder.

Save in datasets/ as	Source
AI_Literacy_questionnaire_data_.csv	Figshare: AI Literacy Questionnaire data
Career_Anxiety_due_to_AI.xlsx	Mendeley: r3xh8sd764
Data_set_Ex.csv (cognitive strain and emotional responses to AI)	Mendeley: 9v9rtzch8g
the cognitive dependence CSV (the file name must contain AI_Cognitive_Dependence)	Mendeley: 7m94b7yr8w

Downloaded files sometimes arrive with different names. Rename them to match the table above.

If your files are somewhere else

Set the environment variable CDAS_FIGSHARE_CSV to the full path of the Figshare CSV.
Or pass the datasets folder to multi_datasets.py: python multi_datasets.py path/to/folder

Optional sanity check before running anything: python audit_datasets.py

Step 3: Run the pipeline

Run the scripts in this order:

#	Command	What it does
1	python data_prep.py	Builds scale scores. Prints sample sizes and correlations.
2	python train_models.py	Trains Ridge, Random Forest and XGBoost on the Figshare data. Saves to results/.
3	python cv_eval.py	Repeated user-grouped cross-validation, a more stable model comparison.
4	python bayes_net.py	Learns the Bayesian network. Saves results/bn_graph.png.
5	python multi_datasets.py	Analyses the other three datasets, including the benefit x cost analysis.

Everything is written to results/.

Expected outputs

After a full run you should have:

Console output from data_prep.py with sample sizes and correlations
Model comparison tables and figures in results/
results/bn_graph.png, the Bayesian network graph
Per-dataset results from multi_datasets.py in results/

Out-of-sample R-squared is reported against a train-mean baseline. Values near zero mean no useful prediction, which is a valid finding, not a bug.

<!-- CHECK: replace the list above with the exact file names your scripts write to results/ -->
Helper scripts

These are for exploring and validating the data. They are not required for the main pipeline.

Script	Purpose	Saved output
audit_datasets.py	Checks that the dataset files are present and well formed	audit_report.txt
inspect_data.py	Lists columns and types	inspect_report.txt
peek_values.py	Shows the values in each column	peek_report.txt
<!-- CHECK: confirm these descriptions match what the scripts do -->

See also DATA_NOTES.md for dataset-specific notes.

Design choices
Anxiety items (A1-A5) are the target and are never used as predictors.
Split by student ID for the Figshare data, so a repeated student never appears in both training and test data.
Baseline comparison: every model is compared with a train-mean baseline.
Missing codes: in the Figshare data, 0 in the course-related scales (IM, S, C, BI) means missing, not a low score.
AI-CogSense (AIAS) survey

A second part of the project, a scoring pipeline and Streamlit dashboard for the AI-CogSense (AIAS) survey, is not yet included in this repository. Only reference material is in cogsense/reference/.

The planned design:

Cleans data: removes duplicate and incomplete rows, flags straight-lining
Scores the AIAS total and its four dimensions as means
Compares students and professionals (Mann-Whitney with FDR adjustment)
Fits a baseline, ElasticNet and Random Forest (with a check model that leaves out C01-C03)
Hides groups smaller than 5 people
Skips the ML step under 60 responses (a 15-30 person pilot is for testing the form, not for modelling)

The AIAS items must be obtained and used under the authors' or publisher's terms. This repo does not contain them. Never use fake or demo data as a result.

Limitations
The public samples are students only. Students vs professionals, privacy concern, locus of control and imposter feelings could not be tested.
Some strong results (for example, regret and fear of weaker thinking) come from near-duplicate questions, not independent constructs.
Results are associations, not causal effects.
Do not collect or store names, emails or phone numbers.
Troubleshooting
Problem	Fix
FileNotFoundError for a dataset	Check the file name and that it is in datasets/. Run python audit_datasets.py.
ModuleNotFoundError	Activate the virtual environment, then pip install -r requirements.txt.
Script can't find the Figshare CSV	Set CDAS_FIGSHARE_CSV to its full path.
pgmpy or xgboost install fails	Upgrade pip (python -m pip install --upgrade pip) and use Python 3.12.
Plots don't appear	They are saved to results/, not shown on screen.