from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
root=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[
md('# Bank Marketing: a CRISP-DM investigation\n**Author: SAM**\n\nThis analysis answers three business questions using the original UCI `bank-full.csv`. AI assistance was used in implementation and writing; all reported outputs are calculated below. Dataset credit: Moro, Rita and Cortez (2014), DOI 10.24432/C5K306; CC BY 4.0.'),
md('## 1. Business understanding\nA bank has limited calling capacity. Which prospects merit attention?\n1. How does subscription rate differ by previous campaign outcome?\n2. How does it differ by number of current-campaign contacts?\n3. Can pre-call information rank likely subscribers in a later holdout period?\n\nThe target is term-deposit subscription, `y=yes`. Ranking metrics are more informative than accuracy because subscriptions are uncommon. No causal or profitability claim is made.'),
code('import pandas as pd\nimport numpy as np\nfrom pathlib import Path\nfrom IPython.display import display, Image\nfrom analysis import load_data, rate_table, fit_model, run\ndf = load_data()\nprint("Shape:", df.shape)\ndisplay(df.head())'),
md('## 2. Data understanding\nThe source file contains 45,211 rows and 16 input fields plus the target, ordered by date. Inputs include customer attributes, last-contact details and prior-campaign information. Audit types, missing values, duplicate rows, special category values and numeric distributions. There are no literal missing cells, but unknown categories and the pdays=-1 sentinel carry missingness semantics.'),
code('display(df.dtypes.to_frame("dtype"))\ndisplay(df.describe(include="all"))\nprint("Missing cells:", int(df.isna().sum().sum()))\nprint("Duplicates:", int(df.duplicated().sum()))\nprint("Positive rate:", df.subscribed.mean())\nprint("Never contacted:", int((df.pdays == -1).sum()))\nprint({c: int((df[c] == "unknown").sum()) for c in df.select_dtypes("object")})'),
md('## 3. Data preparation\nPreserve all source rows: dropping unknown outcomes would discard most observations and distort descriptive rates. For descriptive questions, retain unknown as its own group. For modeling, unknown categorical values become missing, pdays=-1 becomes missing elapsed time with a separate previously_contacted flag. Median numeric imputation and most-frequent categorical imputation, scaling and one-hot encoding are fitted only on training rows.\n\nExclude duration because it is unavailable before the call. Also exclude campaign, day and month because they describe current campaign progression or final contact. The planned contact channel is assumed available at scoring time. Split first 80% for training and last 20% for testing to respect source ordering; do not shuffle. The reusable implementation below shows the full transformations.'),
code('import inspect\nprint(inspect.getsource(fit_model))'),
md('## 4. Analysis: question 1\nDoes prior campaign success identify a high-conversion group? Rates are descriptive, and prior success is a small segment rather than evidence of a causal treatment effect.'),
code('previous = rate_table(df, "poutcome")\ndisplay(previous)\nresults = run()\ndisplay(Image(filename="results/previous_outcome.png"))'),
md('**Answer:** Successful prior outcomes have 64.73% subscription (978/1,511), compared with 9.16% for unknown prior outcomes. Historical engagement appears useful for segmentation; test follow-up experimentally before claiming uplift.'),
md('## 4. Analysis: question 2\nAre more current-campaign contacts associated with more subscriptions? Group into 1, 2–3, 4–5 and 6+ contacts. Counts include the final contact and may reflect difficulty converting a prospect.'),
code('bands = pd.cut(df.campaign, [0,1,3,5,np.inf], labels=["1","2–3","4–5","6+"])\ncontacts = rate_table(df.assign(contact_band=bands), "contact_band")\ndisplay(contacts)\ndisplay(Image(filename="results/contact_frequency.png"))'),
md('**Answer:** Observed rates decline from 14.60% at one contact to 5.81% at six or more. This is an association. Successful customers may stop receiving calls early, so it does not establish that an extra call causes a lower conversion rate. Test a contact cap rather than imposing one solely from this table.'),
md('## 4. Modeling: question 3\nFit logistic regression with balanced class weights for interpretable, scalable ranking. Compare against a most-frequent DummyClassifier. No hyperparameters or thresholds are chosen using test outcomes. Report ROC AUC, average precision, threshold precision/recall and top-20% lift.'),
code('metrics = results["model"]\ndisplay(pd.Series(metrics).to_frame("value"))'),
md('## 5. Evaluation\nA chronological test set shows ROC AUC 0.716 and average precision 0.547. The top-scored 20% has 60.86% observed subscription versus 31.58% in the whole test set, a 1.93× observed lift. The baseline predicts no subscribers and achieves 68.42% accuracy with zero recall. At threshold 0.5 the model has 42.48% precision and 77.66% recall.\n\nThe holdout positive rate is considerably higher than the full-data rate (11.70%), revealing time-period shift. These values describe this historical holdout only. Precision at a budget cutoff does not imply incremental sales or return on investment. There is no customer identifier to test repeat-customer overlap; field availability must be verified before live use.'),
md('## 6. Deployment and communication\nPublish ARTICLE.md/article.html for a nontechnical audience. Run a prospective randomized pilot comparing model-ranked queues with current practice. Verify scoring-time feature availability, current-data performance, subgroup fairness and contact permissions. Measure cost, complaints and actual incremental subscriptions. Select thresholds using capacity and economics.\n\nThe notebook is reusable through documented functions in analysis.py. Included data permit offline reproduction. Sources: https://archive.ics.uci.edu/dataset/222/bank+marketing and original data/bank-names.txt. See README.md for setup and acknowledgments.')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
# Execute real cells through IPython in-process because this container forbids kernel sockets.
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import os, sys
os.chdir(root); sys.path.insert(0, str(root))
shell = InteractiveShell.instance()
count = 0
for cell in nb.cells:
    if cell.cell_type != 'code': continue
    count += 1
    with capture_output() as captured:
        execution = shell.run_cell(cell.source, store_history=True)
    if execution.error_before_exec or execution.error_in_exec:
        raise RuntimeError(execution.error_before_exec or execution.error_in_exec)
    cell.execution_count = count; cell.outputs = []
    if captured.stdout: cell.outputs.append(nbf.v4.new_output('stream', name='stdout', text=captured.stdout))
    if captured.stderr: cell.outputs.append(nbf.v4.new_output('stream', name='stderr', text=captured.stderr))
    for output in captured.outputs:
        cell.outputs.append(nbf.v4.new_output('display_data', data=output.data, metadata=output.metadata))
nb.metadata['execution_note'] = 'All code cells executed sequentially with IPython in-process; network kernel sockets unavailable in build container.' 
nbf.write(nb,root/'Bank_Marketing_CRISP_DM.ipynb')
print('Executed',sum(c.cell_type=='code' for c in nb.cells),'code cells')
try:
 import markdown
 html=markdown.markdown((root/'ARTICLE.md').read_text(),extensions=['tables'])
except ImportError:
 import mistune
 html=mistune.html((root/'ARTICLE.md').read_text())
(root/'article.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Which customers should a bank call first?</title><style>body{max-width:850px;margin:48px auto;padding:0 24px;font:18px/1.65 system-ui;color:#182c39}h1{font-size:46px;line-height:1.15}h2,h3{line-height:1.25;color:#12536c}img{max-width:100%}a{color:#126682}</style></head><body><article>'+html+'</article></body></html>')
