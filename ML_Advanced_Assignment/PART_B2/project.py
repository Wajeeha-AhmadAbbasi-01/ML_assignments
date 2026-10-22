from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

def save_summary(folder, title, lines):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.txt").write_text(title + "\n" + "="*len(title) + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")

def make_data(n=90, p=300, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # Correlated blocks mimic related gene-expression measurements.
    for start, end in [(20, 40), (80, 100)]:
        shared = rng.normal(size=n)
        X[:, start:end] = shared[:, None] + rng.normal(scale=0.25, size=(n, end-start))
    signal_idx = [2, 7, 15, 28, 34, 65, 88, 93, 140, 222]
    beta = np.zeros(p)
    beta[signal_idx] = [3.0, -2.8, 2.5, 2.2, -2.0, 1.8, -2.5, 2.1, 1.7, -1.6]
    y = X @ beta + rng.normal(scale=2.5, size=n)
    return X, y, [f"gene_{i+1}" for i in range(p)], signal_idx

def pipe(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])

def q1(X, y):
    folder = OUT/"Q1_OLS_p_greater_n"
    folder.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=SEED)
    models = {"OLS": pipe(LinearRegression()), "Ridge (alpha=10)": pipe(Ridge(alpha=10))}
    rows=[]
    for name, model in models.items():
        model.fit(Xtr,ytr)
        rows.append({"model":name,"train_MSE":mean_squared_error(ytr,model.predict(Xtr)),
                     "test_MSE":mean_squared_error(yte,model.predict(Xte)),
                     "train_R2":r2_score(ytr,model.predict(Xtr)),"test_R2":r2_score(yte,model.predict(Xte))})
    df=pd.DataFrame(rows); df.to_csv(folder/"ols_vs_ridge_metrics.csv",index=False)
    plt.figure(figsize=(7,5)); x=np.arange(2); w=.35
    plt.bar(x-w/2,df.train_MSE,w,label="Train MSE"); plt.bar(x+w/2,df.test_MSE,w,label="Test MSE")
    plt.xticks(x,df.model); plt.ylabel("Mean squared error"); plt.title("OLS overfitting in p > n data"); plt.legend(); plt.tight_layout()
    plt.savefig(folder/"ols_overfitting.png",dpi=160); plt.close()
    rank=np.linalg.matrix_rank(Xtr)
    save_summary(folder,"Q1 — OLS in the p > n regime",[
        f"Dataset: n={len(X)} samples, p={X.shape[1]} features; p > n.",
        f"Training split has {len(Xtr)} rows and {Xtr.shape[1]} features; design-matrix rank={rank}.",
        f"OLS train MSE={df.iloc[0].train_MSE:.4f}; OLS test MSE={df.iloc[0].test_MSE:.4f}.",
        f"Ridge train MSE={df.iloc[1].train_MSE:.4f}; Ridge test MSE={df.iloc[1].test_MSE:.4f}.",
        "OLS can return a least-squares solution even when p > n, but coefficients are not uniquely determined and can be unstable. A near-perfect training fit does not guarantee good test performance.",
        "Ridge adds an L2 penalty that stabilizes coefficients. Compare held-out test MSE rather than judging only by training fit."
    ])
    print("Q1 complete")

def q2(X,y):
    folder=OUT/"Q2_Nested_CV_Comparison"
    folder.mkdir(parents=True, exist_ok=True)
    outer=KFold(n_splits=5,shuffle=True,random_state=SEED)
    inner=KFold(n_splits=4,shuffle=True,random_state=SEED+1)
    grids={
        "Ridge":(Ridge(),{"model__alpha":np.logspace(-2,3,12)}),
        "Lasso":(Lasso(max_iter=30000),{"model__alpha":np.logspace(-3,1,12)}),
        "ElasticNet":(ElasticNet(max_iter=30000),{"model__alpha":np.logspace(-3,1,10),"model__l1_ratio":[.1,.3,.5,.7,.9]})
    }
    rows=[]
    for name,(est,grid) in grids.items():
        for fold,(itr,ite) in enumerate(outer.split(X),1):
            search=GridSearchCV(pipe(est),grid,scoring="neg_mean_squared_error",cv=inner,n_jobs=-1)
            search.fit(X[itr],y[itr]); pred=search.predict(X[ite])
            rows.append({"model":name,"outer_fold":fold,"test_MSE":mean_squared_error(y[ite],pred),
                         "test_R2":r2_score(y[ite],pred),"best_params":str(search.best_params_),
                         "inner_best_MSE":-search.best_score_})
    df=pd.DataFrame(rows); df.to_csv(folder/"nested_cv_fold_results.csv",index=False)
    summary=df.groupby("model").agg(mean_outer_MSE=("test_MSE","mean"),std_outer_MSE=("test_MSE","std"),
        mean_outer_R2=("test_R2","mean"),std_outer_R2=("test_R2","std")).reset_index()
    summary.to_csv(folder/"nested_cv_model_summary.csv",index=False)
    plt.figure(figsize=(8,5)); plt.bar(summary.model,summary.mean_outer_MSE,yerr=summary.std_outer_MSE,capsize=4)
    plt.ylabel("Outer-fold MSE (lower is better)"); plt.title("Nested CV comparison"); plt.tight_layout()
    plt.savefig(folder/"nested_cv_comparison.png",dpi=160); plt.close()
    lines=["Outer loop: 5 folds; inner tuning loop: 4 folds.",
           "Scaling is inside the pipeline, so it is fitted only on each training fold.",
           "Outer scores estimate generalization; inner scores are used to choose hyperparameters.","Mean outer-fold results:"]
    for _,r in summary.iterrows():
        lines.append(f"{r.model}: MSE={r.mean_outer_MSE:.4f} (SD={r.std_outer_MSE:.4f}); R2={r.mean_outer_R2:.4f} (SD={r.std_outer_R2:.4f}).")
    best=summary.loc[summary.mean_outer_MSE.idxmin()]
    lines.append(f"Lowest mean outer-fold MSE in this run: {best.model}.")
    save_summary(folder,"Q2 — Nested CV comparison",lines)
    print("Q2 complete")

def q3(X,y,names,signals):
    folder=OUT/"Q3_Bootstrap_Feature_Stability"
    folder.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(2026); B=80; counts=np.zeros(X.shape[1],int); rows=[]
    for b in range(B):
        idx=rng.integers(0,len(X),size=len(X))
        model=pipe(Lasso(alpha=.08,max_iter=30000)); model.fit(X[idx],y[idx])
        coef=model.named_steps["model"].coef_; selected=np.abs(coef)>1e-6
        counts += selected
        for j,name in enumerate(names):
            rows.append({"bootstrap":b+1,"feature":name,"coefficient":coef[j],
                         "selected":bool(selected[j]),"is_true_signal":j in signals})
    stability=pd.DataFrame({"feature":names,"selection_count":counts,"selection_frequency":counts/B,
                            "is_true_signal":[i in signals for i in range(len(names))]})
    stability=stability.sort_values(["selection_frequency","feature"],ascending=[False,True])
    stability.to_csv(folder/"bootstrap_feature_stability.csv",index=False)
    pd.DataFrame(rows).to_csv(folder/"bootstrap_coefficients.csv",index=False)
    top=stability.head(25).sort_values("selection_frequency")
    plt.figure(figsize=(9,8)); plt.barh(top.feature,top.selection_frequency)
    plt.xlabel("Selection frequency across 80 bootstrap samples"); plt.ylabel("Feature")
    plt.title("Top 25 features by Lasso selection frequency"); plt.xlim(0,1); plt.tight_layout()
    plt.savefig(folder/"bootstrap_stability_top25.png",dpi=160); plt.close()
    sig=stability[stability.is_true_signal].selection_frequency
    noise=stability[~stability.is_true_signal].selection_frequency
    stable=stability[stability.is_true_signal & (stability.selection_frequency>=.7)].feature.tolist()
    save_summary(folder,"Q3 — Bootstrap feature-selection stability",[
        f"Bootstrap resamples: {B}; each sample is drawn with replacement and contains {len(X)} rows.",
        "Lasso alpha fixed at 0.08 to make selection frequencies comparable across resamples.",
        f"Known signal features selected in at least 70% of resamples: {', '.join(stable) if stable else 'None'}.",
        f"Mean selection frequency for known signal features: {sig.mean():.3f}.",
        f"Mean selection frequency for noise features: {noise.mean():.3f}.",
        "High selection frequency suggests robustness to changes in sampled observations. Correlated predictors can substitute for each other, so individual feature stability may be low even when a group carries signal.",
        "Stability is not proof of causality."
    ])
    print("Q3 complete")

def q4():
    folder=OUT/"Q4_Final_Recommendation"
    folder.mkdir(parents=True, exist_ok=True)
    scores=pd.read_csv(OUT/"Q2_Nested_CV_Comparison"/"nested_cv_model_summary.csv")
    stability=pd.read_csv(OUT/"Q3_Bootstrap_Feature_Stability"/"bootstrap_feature_stability.csv")
    best=scores.loc[scores.mean_outer_MSE.idxmin()]
    high=stability[stability.selection_frequency>=.7]
    true_high=int(high.is_true_signal.sum()); noise_high=len(high)-true_high
    if best.model=="Lasso":
        rec="Lasso is the leading predictive choice in this run and is useful when a compact, interpretable feature set is the goal. Verify feature stability before trusting individual selected predictors."
    elif best.model=="ElasticNet":
        rec="ElasticNet is the leading predictive choice in this run. It combines sparse selection with an L2 component that can share weight among correlated predictors, making it a strong default when correlated feature groups are expected."
    else:
        rec="Ridge is the leading predictive choice in this run. Prefer it when prediction is the priority and many features may carry signal, because it stabilizes coefficients without forcing them to zero. It does not perform hard feature selection."
    lines=[
        "This recommendation is based on this synthetic dataset; results may differ on real data.",
        f"Lowest mean outer-fold MSE: {best.model} (MSE={best.mean_outer_MSE:.4f}; R2={best.mean_outer_R2:.4f}).",
        f"Features selected in >=70% of bootstrap samples: {len(high)} total; {true_high} known signal and {noise_high} noise features.",
        "FINAL RECOMMENDATION:",rec,"",
        "Rule of thumb:",
        "- Ridge: prediction with many potentially useful predictors; keeps all coefficients.",
        "- Lasso: sparse model and feature selection; may choose one predictor from a correlated group.",
        "- ElasticNet: sparse model when predictors are strongly correlated; often keeps groups together.",
        "For real data, keep feature selection and preprocessing inside training folds and evaluate on held-out data. Selection frequency does not establish causality."
    ]
    save_summary(folder,"Q4 — Final recommendation",lines)
    (OUT/"final_recommendation.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Q4 complete")

def main():
    X,y,names,signals=make_data()
    pd.DataFrame([{"n_samples":len(X),"n_features":X.shape[1],"p_greater_than_n":X.shape[1]>len(X),
                   "true_signal_count":len(signals),"random_state":SEED}]).to_csv(OUT/"dataset_description.csv",index=False)
    q1(X,y); q2(X,y); q3(X,y,names,signals); q4()
    summaries=[]
    for name in ["Q1_OLS_p_greater_n","Q2_Nested_CV_Comparison","Q3_Bootstrap_Feature_Stability","Q4_Final_Recommendation"]:
        p=OUT/name/"summary.txt"
        if p.exists(): summaries.append(p.read_text(encoding="utf-8"))
    (OUT/"overall_summary.txt").write_text("SPARSE MODELING FOR HIGH-DIMENSIONAL DATA — OVERALL SUMMARY\n"+"="*59+"\n\n"+"\n\n".join(summaries),encoding="utf-8")
    print(f"\nAll questions complete. Outputs saved in {OUT}")

if __name__=="__main__":
    main()
