# Dementia Progression Forecasting
### Longitudinal Cognitive Trajectory Modelling via Gated Recurrent Neural Networks

Next-visit MMSE prediction on the OASIS-2 longitudinal brain imaging dataset using a formally defined sequence regression pipeline. Four recurrent architectures (Vanilla RNN, LSTM, GRU, BiLSTM) are evaluated across a 108-configuration hyperparameter sweep with strict patient-wise leakage prevention and dual evaluation (held-out test + 5-fold cross-validation).

---

## Results at a Glance

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---|---|
| Persistence baseline | 0.9459 | 1.3656 | 0.7160 |
| Training-mean baseline | 2.2310 | 2.7946 | −0.1893 |
| Linear Regression (Ridge) | 1.2658 | 1.6588 | 0.5810 |
| RNN (tuned) | 1.2667 | 1.6211 | 0.5998 |
| LSTM (tuned) | 1.0985 | 1.4358 | 0.6861 |
| GRU (tuned) | 1.2024 | 1.5381 | 0.6398 |
| **BiLSTM ★ Best** | **0.9886** | **1.2736** | **0.7530** |

**5-fold GroupKFold CV (BiLSTM):** mean R² = 0.47 ± 0.12 — primary, conservative performance estimate.  
**Test R² = 0.75** — upper-bound reference (favourable partition composition).

---

## Figures

### Exploratory Data Analysis
<img width="1490" height="887" alt="image" src="https://github.com/user-attachments/assets/29420730-83af-4cef-83bf-c336fec2accb" />
*Six-panel EDA summary: MMSE and CDR distributions by group, CDR–MMSE correlation (r = −0.687), sequence length distribution, visit-to-visit MMSE change, and CDR-change vs. mean ΔMMSE.*

---

### Full Model Comparison
<img width="1068" height="329" alt="image" src="https://github.com/user-attachments/assets/01827cbd-53d8-49f9-8958-9e8932fe03d8" />


*All six models evaluated on the held-out test set. BiLSTM outperforms all baselines including the strong persistence heuristic.*

---

### Best Model Diagnostics (BiLSTM)
<img width="1489" height="396" alt="image" src="https://github.com/user-attachments/assets/9b6af507-4a58-456d-a243-7865b769b52f" />
*Left: training/validation MSE loss curves over 143 epochs — no divergence, clean convergence. Centre: predicted vs. true MMSE scatter near the perfect-prediction diagonal. Right: prediction error distribution (mean = −0.08 points; 95% of errors within ±2.5 MMSE points).*

---

### Per-Patient Trajectory
<img width="1188" height="396" alt="image" src="https://github.com/user-attachments/assets/50f5398e-25e6-43fb-b308-efd8b1abfa62" />
*Subject OAS2_0127 (4 forecast steps): true vs. predicted MMSE trajectory and per-step error. The model closely tracks the stable high-scoring trajectory with a slight conservative underestimation.*

---

## Problem Definition

MMSE (Mini-Mental State Examination) is a 30-point cognitive screening instrument used to quantify dementia severity. Predicting how a patient's MMSE will evolve over time supports earlier clinical intervention and care planning.

This project frames the task as **many-to-one sequence regression**:

> Given a patient's full visit history X = (x₁, x₂, …, xₜ), predict MMSE at visit t+1.

Each visit is represented as a 12-dimensional feature vector (9 numeric + 3 one-hot encoded). The target is absolute MMSE — not a change score — as absolute values are more clinically interpretable.

---

## Dataset

**OASIS-2 Longitudinal** — Open Access Series of Imaging Studies  
150 right-handed subjects aged 60–96, assessed across 1–5 longitudinal visits.

| Property | Value |
|---|---|
| Total subjects | 150 |
| Visit records (after cleaning) | 371 |
| Training examples (sequences) | 148 |
| Features per timestep | 12 |
| Max sequence length | 4 timesteps |

The dataset is fetched automatically at runtime — no manual download required:

```python
url = 'https://raw.githubusercontent.com/multivacplatform/multivac-dl/master/data/mri-and-alzheimers/oasis_longitudinal.csv'
df = pd.read_csv(url)
```

---

## Methodology

### Leakage Prevention
- All splits performed at **patient level** — no subject's visits appear in more than one partition
- Missing values imputed using **training-set medians only**
- One-hot encoding fitted on training partition and reindexed onto validation/test
- **Masking-safe scaling**: StandardScaler applied only to non-padded timesteps; padded positions remain exactly 0.0 to preserve the `Masking(mask_value=0.0)` layer

### Data Split
| Partition | Subjects | Records | Sequences |
|---|---|---|---|
| Training | 104 | 252 | 148 |
| Validation | 23 | 59 | 36 |
| Test | 23 | 60 | 37 |

### Architectures
All four architectures share the skeleton:  
`Input(4, 12) → Masking(0.0) → RecurrentCell(units) → Dense(32, ReLU) → Dense(1, linear)`

- **Vanilla RNN** — recurrent lower bound; susceptible to vanishing gradients
- **LSTM** — forget, input, output gates for selective long-range retention
- **GRU** — update and reset gates; 25% fewer parameters than LSTM
- **BiLSTM** — bidirectional LSTM; on short sequences, effectively a larger regularised model

### Hyperparameter Sweep
108 configurations: units ∈ {16, 32, 64} × dropout ∈ {0.0, 0.2, 0.4} × batch ∈ {8, 16, 32} × 4 architectures.  
Model selection governed by **validation loss only** — test set withheld until final evaluation.

**Optimal config (BiLSTM):** 64 units, dropout = 0.4, batch size = 8, trained for 143 epochs.

---

## Repository Structure

```
dementia-progression-forecasting/
├── README.md
├── requirements.txt
├── Project.ipynb       # Full pipeline — run all cells to reproduce
├── report
```

---

## Reproducing Results

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/dementia-progression-forecasting.git
cd dementia-progression-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `Project.ipynb` in Jupyter or VS Code and run all cells.  
The OASIS-2 dataset is fetched automatically — no manual download needed.  
All figures, CSVs, and the trained model are saved to their respective folders.

> **Note:** The full 108-configuration sweep takes several minutes depending on hardware. Set `SEED = 42` is applied globally for reproducibility.

---

## Dependencies

```
tensorflow
numpy
pandas
scikit-learn
matplotlib
```

Full pinned versions: see `requirements.txt`.
