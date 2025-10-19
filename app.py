import streamlit as st, pandas as pd, joblib
st.title('Credit Scoring - Synthetic Demo')
st.write('Train creates a synthetic dataset and trains a GradientBoosting model.')
if st.button('Train model (creates data/credit_synthetic.csv)'):
    import subprocess, sys
    subprocess.run([sys.executable, 'src/train.py'])
    st.success('Training done.')
if st.button('Load model'):
    st.session_state['credit_pack'] = joblib.load('models/credit_model.joblib')
    st.success('Model loaded.')
if st.button('Show feature importances') and 'credit_pack' in st.session_state:
    import pandas as pd
    pack = st.session_state['credit_pack']
    model = pack['model']
    cols = pack['columns']
    importances = model.feature_importances_
    df = pd.DataFrame({'feature':cols, 'importance': importances}).sort_values('importance', ascending=False)
    st.table(df.head(10))
st.write('You can upload a CSV with same columns or fill a single sample:')
if 'credit_pack' in st.session_state:
    cols = st.session_state['credit_pack']['columns']
    sample = {c: 0 for c in cols}
    st.write('Sample input:')
    vals = {}
    for c in cols:
        vals[c] = st.number_input(c, value=float(sample[c]))
    if st.button('Predict sample') :
        import numpy as np
        model = st.session_state['credit_pack']['model']
        scaler = st.session_state['credit_pack']['scaler']
        import pandas as pd
        df = pd.DataFrame([vals])[cols]
        Xs = scaler.transform(df)
        pred = model.predict(Xs)[0]
        prob = model.predict_proba(Xs)[0,1]
        st.write('Predicted good(1)/bad(0):', int(pred), 'probability of positive(good):', float(prob))
else:
    st.info('Load model first.')
