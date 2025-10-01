# Report: Accident Risk Prediction (CNN-BiLSTM-Attn + DeepSHAP)

## Datasets
- UK (2005–2018): counts, splits
- US (2016–2020): counts, splits

## Preprocessing
- Removed >10% missing features; near-zero variance threshold=0.02
- Temporal parts: year, month, week, hour
- Encoders: LabelEncoder per categorical
- 32 indicators (Table 1): [TO BE FILLED]
- Target construction: [DETAILS FROM PAPER]

## Model
- CNN-BiLSTM-Attention with spatial+temporal local attention
- Hyperparameters (Table 3): [TO BE FILLED]

## Training
- Chronological split 6:2:2, 10-fold CV (training robustness)
- SMOTE + RandomUnderSampler on training windows
- Weighted MSE (1/2/3 risk weighting)

## Results
- MAE (UK): [value]; Paper: 0.2475
- MAE (US): [value]; Paper: 0.2683
- Classification (rounded to 1/2/3): P/R/F1 = [values]
- Curves and plots: see `outputs/`

## Explainability (DeepSHAP)
- Global feature ranking (mean|SHAP|): top 15
- Retrain with top-15: metrics vs full features

## Notes / Deviations
- Any differences from paper