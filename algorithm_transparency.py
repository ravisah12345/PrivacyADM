"""
COM6020M - Privacy and Data Protection
Appendix: Algorithm Transparency in Automated Decision-Making

This script demonstrates algorithm transparency techniques
applied to a simulated loan approval ADM system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# STEP 1 — Simulate a loan application dataset
np.random.seed(42)
n_samples = 500

data = {
    'age':              np.random.randint(18, 70, n_samples),
    'annual_income':    np.random.randint(15000, 120000, n_samples),
    'credit_score':     np.random.randint(300, 850, n_samples),
    'employment_years': np.random.randint(0, 30, n_samples),
    'existing_debt':    np.random.randint(0, 50000, n_samples),
    'loan_amount':      np.random.randint(1000, 50000, n_samples),
}

df = pd.DataFrame(data)

df['approved'] = (
    (df['credit_score'] > 600) &
    (df['annual_income'] > 30000) &
    (df['existing_debt'] < 20000)
).astype(int)

noise_idx = np.random.choice(df.index, size=40, replace=False)
df.loc[noise_idx, 'approved'] = 1 - df.loc[noise_idx, 'approved']

print("=" * 60)
print("  COM6020M — Algorithm Transparency Demonstration")
print("  Loan Approval ADM System")
print("=" * 60)
print(f"\nDataset: {len(df)} loan applications")
print(f"Approved: {df['approved'].sum()} ({df['approved'].mean()*100:.1f}%)")
print(f"Rejected: {(1-df['approved']).sum()} ({(1-df['approved']).mean()*100:.1f}%)")

# STEP 2 — Train an interpretable Decision Tree model
features = ['age', 'annual_income', 'credit_score',
            'employment_years', 'existing_debt', 'loan_amount']
X = df[features]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'─'*60}")
print("  Model Performance")
print(f"{'─'*60}")
print(f"Accuracy: {accuracy*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Rejected', 'Approved']))

# STEP 3 — Feature Importance Analysis (XAI)
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\n{'─'*60}")
print("  Feature Importance — What Drives Decisions?")
print(f"{'─'*60}")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['Importance'] * 40)
    print(f"  {row['Feature']:20s} {bar} {row['Importance']:.3f}")

print(f"\n  PRIVACY NOTE: '{importance_df.iloc[0]['Feature']}' is the")
print(f"  most influential factor. If this attribute correlates")
print(f"  with protected characteristics (e.g. ethnicity, health),")
print(f"  the model may perpetuate discriminatory outcomes —")
print(f"  a core concern of inferential profiling (Section 2.3).")

# STEP 4 — Per-Decision Explanation
print(f"\n{'─'*60}")
print("  Per-Decision Explanation (GDPR Article 22 Compliance)")
print(f"{'─'*60}")

applicants = pd.DataFrame({
    'age':              [35, 28],
    'annual_income':    [55000, 22000],
    'credit_score':     [720, 480],
    'employment_years': [8, 1],
    'existing_debt':    [5000, 25000],
    'loan_amount':      [10000, 15000],
})

applicant_names = ["Applicant A", "Applicant B"]
decisions = model.predict(applicants)
probabilities = model.predict_proba(applicants)

for i, name in enumerate(applicant_names):
    decision = "APPROVED" if decisions[i] == 1 else "REJECTED"
    confidence = probabilities[i][decisions[i]] * 100
    print(f"\n  {name}: {decision} (confidence: {confidence:.1f}%)")
    print(f"  Profile:")
    for feat in features:
        print(f"    {feat:20s}: {applicants.iloc[i][feat]:,}")

    node_indicator = model.decision_path(applicants.iloc[[i]])
    node_ids = node_indicator.indices
    feature_idx = model.tree_.feature
    thresholds = model.tree_.threshold

    print(f"  Key decision factors:")
    seen = set()
    for node_id in node_ids:
        feat_i = feature_idx[node_id]
        if feat_i >= 0 and feat_i not in seen:
            feat_name = features[feat_i]
            threshold = thresholds[node_id]
            value = applicants.iloc[i][feat_name]
            direction = "above" if value > threshold else "at/below"
            print(f"    - {feat_name}: {value:,} ({direction} threshold of {threshold:,.0f})")
            seen.add(feat_i)

# STEP 5 — Visualisations
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    'COM6020M — Algorithm Transparency in ADM Systems\n'
    'Loan Approval Decision Tree Analysis',
    fontsize=13, fontweight='bold', y=1.02
)

colours = ['#C0392B' if imp == importance_df['Importance'].max()
           else '#2E5496' for imp in importance_df['Importance']]
bars = axes[0].barh(
    importance_df['Feature'],
    importance_df['Importance'],
    color=colours, edgecolor='white', linewidth=0.5
)
axes[0].set_xlabel('Feature Importance Score', fontsize=11)
axes[0].set_title('Feature Importance\n(Which attributes drive decisions?)', fontsize=11)
axes[0].invert_yaxis()
axes[0].set_xlim(0, importance_df['Importance'].max() * 1.3)

for bar, val in zip(bars, importance_df['Importance']):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)

top_feature = importance_df.iloc[0]['Feature']
axes[0].annotate(
    f'Most influential:\n"{top_feature}"\nPrivacy risk if correlated\nwith protected attributes',
    xy=(importance_df.iloc[0]['Importance'], 0),
    xytext=(importance_df['Importance'].max() * 0.7, 2.5),
    fontsize=8, color='#C0392B',
    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.2)
)

plot_tree(
    model,
    feature_names=features,
    class_names=['Rejected', 'Approved'],
    filled=True,
    rounded=True,
    fontsize=7,
    ax=axes[1],
    impurity=False,
    precision=0
)
axes[1].set_title('Decision Tree Structure\n(Full decision logic — human readable)', fontsize=11)

approved_patch = mpatches.Patch(color='#85B7EB', label='Approved path')
rejected_patch = mpatches.Patch(color='#F09595', label='Rejected path')
axes[1].legend(handles=[approved_patch, rejected_patch],
               loc='lower right', fontsize=8)

plt.tight_layout()

# ── FIXED: Save to your Desktop ──────────────────────────
plt.savefig('/Users/ravisah/Desktop/algorithm_transparency.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n{'─'*60}")
print("  Visualisation saved to Desktop: algorithm_transparency.png")
print(f"{'─'*60}")

# STEP 6 — Privacy Risk Summary
print(f"\n{'─'*60}")
print("  Privacy Risk Assessment Summary")
print(f"{'─'*60}")
print("""
  This demonstration highlights three key privacy concerns:

  1. DATA MINIMISATION (Section 2.2 / GDPR Article 5)
     The model uses 6 personal attributes. A privacy-
     preserving deployment should assess whether all 6
     are strictly necessary.

  2. TRANSPARENCY (Section 2.1 / GDPR Article 22)
     The Decision Tree provides human-readable decision
     paths — this is the principle behind XAI.

  3. INFERENTIAL RISK (Section 2.3)
     If credit_score correlates with protected attributes
     such as ethnicity or postcode, the model may indirectly
     discriminate. Algorithmic auditing (Recommendation 1)
     is essential to detect this.
""")

print("  Script complete.")
print("=" * 60)
