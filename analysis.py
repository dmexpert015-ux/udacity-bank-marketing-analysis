"""Reproduce three marketing questions using the unmodified UCI bank-full data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent

def load_data():
    """Read the semicolon-delimited full dataset; preserve source row ordering."""
    df = pd.read_csv(ROOT / 'data/bank-full.csv', sep=';')
    assert len(df) == 45211 and set(df.y.unique()) == {'yes', 'no'}
    assert df.duplicated().sum() == 0
    df['subscribed'] = (df.y == 'yes').astype(int)
    return df

def rate_table(df, column):
    """Return sample counts and observed subscription rates by a group."""
    return df.groupby(column, observed=True).subscribed.agg(['count', 'sum', 'mean'])

def fit_model(df):
    """Train only on early rows; exclude call duration and same-campaign outcomes."""
    # Day/month describe the last contact rather than prospect attributes. Exclude
    # them, campaign counts, and duration for a conservative pre-campaign model.
    X = df.drop(columns=['y', 'subscribed', 'duration', 'campaign', 'day', 'month'])
    X = X.replace('unknown', np.nan)
    # -1 means never contacted, not a negative elapsed duration. Preserve that
    # information separately before imputing the elapsed-time value.
    X['previously_contacted'] = (X.pdays != -1).astype(int)
    X['pdays'] = X.pdays.replace(-1, np.nan)
    y = df.subscribed
    split = int(len(df) * .8)
    train_X, test_X, train_y, test_y = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
    categorical = X.select_dtypes(include='object').columns.tolist()
    numeric = X.select_dtypes(exclude='object').columns.tolist()
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical)
    ])
    model = Pipeline([('prepare', prep), ('classify', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42))])
    model.fit(train_X, train_y)
    probabilities = model.predict_proba(test_X)[:, 1]
    baseline = DummyClassifier(strategy='most_frequent').fit(train_X, train_y)
    k = int(np.ceil(.2 * len(test_y)))
    selected = np.argsort(probabilities)[-k:]
    metrics = {
        'train_rows': len(train_y), 'test_rows': len(test_y),
        'test_subscription_rate': float(test_y.mean()),
        'roc_auc': float(roc_auc_score(test_y, probabilities)),
        'average_precision': float(average_precision_score(test_y, probabilities)),
        'precision_at_0_5': float(precision_score(test_y, probabilities >= .5)),
        'recall_at_0_5': float(recall_score(test_y, probabilities >= .5)),
        'top_20_percent_rows': k,
        'top_20_percent_subscription_rate': float(test_y.iloc[selected].mean()),
        'top_20_percent_lift': float(test_y.iloc[selected].mean() / test_y.mean()),
        'baseline_accuracy': float(baseline.score(test_X, test_y)),
        'baseline_recall': float(recall_score(test_y, baseline.predict(test_X), zero_division=0)),
        'excluded_columns': ['duration', 'campaign', 'day', 'month']
    }
    return metrics

def run():
    """Write auditable tables, charts and model metrics; return all findings."""
    df = load_data()
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    previous = rate_table(df, 'poutcome')
    bands = pd.cut(df.campaign, [0,1,3,5,np.inf], labels=['1','2–3','4–5','6+'])
    campaign = rate_table(df.assign(contact_band=bands), 'contact_band')
    previous.to_csv(out / 'previous_outcome.csv')
    campaign.to_csv(out / 'contact_frequency.csv')
    for table, name, title in [(previous, 'previous_outcome', 'Subscription rate by previous campaign outcome'), (campaign, 'contact_frequency', 'Subscription rate by current campaign contact count')]:
        fig, ax = plt.subplots(figsize=(8,4.6))
        ax.bar(table.index.astype(str), table['mean']*100, color='#146b82')
        ax.set_ylabel('Observed subscription rate (%)')
        ax.set_title(title)
        for i, (_, row) in enumerate(table.iterrows()):
            ax.text(i, row['mean']*100+.5, f"{row['mean']:.1%}\nn={int(row['count']):,}", ha='center', fontsize=9)
        ax.set_ylim(0, table['mean'].max()*100+12)
        fig.tight_layout(); fig.savefig(out / (name+'.png'), dpi=180); plt.close(fig)
    metrics = fit_model(df)
    results = {'rows': len(df), 'subscribers': int(df.subscribed.sum()), 'overall_rate': float(df.subscribed.mean()), 'missing_cells': int(df.isna().sum().sum()), 'unknown_counts': {c:int((df[c]=='unknown').sum()) for c in df.select_dtypes('object')}, 'model':metrics}
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results

if __name__ == '__main__':
    run()
