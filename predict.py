import joblib, pandas as pd, numpy as np
def load_model(path='models/credit_model.joblib'):
    data = joblib.load(path)
    return data['model'], data['scaler'], data['columns']
def predict(model, scaler, cols, X_df):
    import pandas as pd, numpy as np
    if isinstance(X_df, dict):
        X_df = pd.DataFrame([X_df])
    elif isinstance(X_df, list):
        X_df = pd.DataFrame(X_df)
    X_df = X_df[cols]
    Xs = scaler.transform(X_df)
    prob = model.predict_proba(Xs)[:,1]
    pred = model.predict(Xs)
    return pred, prob
