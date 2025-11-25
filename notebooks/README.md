# XAI Analysis Notebooks

This directory contains the complete XAI (Explainable AI) analysis for the credit risk thesis.

## 📁 Notebook Overview

### Data Exploration
- **`00_data_exploration.ipynb`** - Descriptive statistics and distributions for thesis appendix

### XAI Analysis (Thesis Chapters)
- **`05a_xai_global.ipynb`** - Chapter 1: Global Model Understanding (✅ Complete)
- **`05b_xai_temporal.ipynb`** - Chapter 2: Temporal Evolution (📋 Ready to run)
- **`05c_xai_case_studies.ipynb`** - Chapter 3: Case Studies (📋 Ready to run)

### Legacy
- **`05_xai_exploration.ipynb`** - Original notebook (superseded by 05a-05c)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you're in the project virtual environment
source .venv/bin/activate  # or activate.bat on Windows

# All dependencies should already be installed
```

### Running the Analysis

**Execute notebooks in order:**

```bash
# Step 1: Data exploration (optional - for appendix)
jupyter notebook 00_data_exploration.ipynb

# Step 2: Already complete - review if needed
jupyter notebook 05a_xai_global.ipynb

# Step 3: Temporal analysis (MAIN CONTRIBUTION)
jupyter notebook 05b_xai_temporal.ipynb

# Step 4: Case studies
jupyter notebook 05c_xai_case_studies.ipynb
```

**Expected runtime:**
- `00_data_exploration.ipynb`: ~5 minutes
- `05a_xai_global.ipynb`: Already complete (~10 min if rerun)
- `05b_xai_temporal.ipynb`: ~30-60 minutes (trains 5 models)
- `05c_xai_case_studies.ipynb`: ~10 minutes

**Total: ~1-2 hours**

---

## 📊 What Each Notebook Does

### 00_data_exploration.ipynb
**Purpose**: Generate descriptive statistics for thesis appendix

**Outputs**:
- Dataset summary table
- Feature distribution histograms
- Default rate evolution plot
- SME size and industry analysis
- Feature correlation matrix

**Files created**:
- `results/data_exploration/feature_statistics.csv`
- `results/data_exploration/dataset_summary.csv`

---

### 05a_xai_global.ipynb (Chapter 1)
**Research Question**: What drives SME bankruptcy risk?

**Key Analyses**:
1. Train LightGBM (AUC 0.948) vs Logistic (AUC 0.869)
2. Feature importance comparison (SHAP + gain)
3. ALE plots revealing non-linear relationships
4. SHAP interaction analysis

**Key Finding**: **Interactions are weak** (<30% of main effects)
- Model is predominantly additive
- Binary thresholds (zero vs non-zero) drive performance
- Proxy features identified (dividend_yield)

**Figures Generated**:
- FIGURE 1: Performance comparison table
- FIGURE 2: ALE plots (15 features)
- FIGURE 3A/B: SHAP summary (beeswarm)
- FIGURE 4: SHAP interaction heatmap

**Files created**:
- `results/xai_exploration/shap_cache.pkl` (70.4 MB)

---

### 05b_xai_temporal.ipynb (Chapter 2) ⭐ MAIN CONTRIBUTION
**Research Question**: Do risk relationships change across economic regimes?

**Economic Periods Analyzed** (2008-2023):
1. Financial Crisis (2008-2010)
2. Sovereign Debt (2011-2013)
3. Recovery (2014-2018)
4. COVID (2020-2021)
5. Post-COVID (2022-2023)

**Key Analyses**:
1. Train separate model for each period (on preceding data)
2. Feature importance evolution heatmap
3. ALE plot evolution (threshold shifts)
4. Model performance stability

**Expected Findings**:
- Behavioral features (dividends, events) remain stable
- Liquidity becomes 2× more important during crises
- "Safe" thresholds shift upward (e.g., 15% → 25% cash ratio)
- Models remain accurate across regimes (relationships stable)

**Figures Generated**:
- FIGURE 5: Importance rank heatmap (features × periods)
- FIGURE 6: ALE evolution plots
- Performance by period table

**Files created**:
- `results/xai_temporal/importance_evolution.csv`
- `results/xai_temporal/performance_by_period.csv`
- `results/xai_temporal/temporal_cache.pkl`

---

### 05c_xai_case_studies.ipynb (Chapter 3)
**Research Question**: When and why does LightGBM outperform Logistic?

**Key Analyses**:
1. Identify model disagreements (|ΔPD| > 20%)
2. Characterize firm types where ML adds value
3. Select 6-8 representative case studies:
   - LightGBM caught, Logit missed (ML advantage)
   - Logit caught, LightGBM missed (linear patterns)
   - Both correct (clear cases)
   - Both failed (model limitations)
   - False alarms

4. Generate SHAP waterfall plots (individual explanations)
5. Write plain-language narratives

**Expected Findings**:
- ML adds value for **moderate-risk, complex profiles**
- Younger, smaller firms benefit most
- Behavioral signals + binary thresholds detected
- Clear linear cases don't need ML
- SHAP explanations are regulatory-suitable

**Figures Generated**:
- 6-8 SHAP waterfall plots (case studies)
- Model disagreement summary table
- Firm characteristics comparison

---

## 🎯 Strategic Decisions

### Why No Clustering?
**Original plan**: SHAP-based clustering to identify "default taxonomies"

**Decision**: **Excluded** based on weak interaction finding (Chapter 1)

**Rationale**:
- Weak interactions → additive model behavior
- Clustering would just show "which main effects dominate" (not insightful)
- Better to focus on temporal evolution (main contribution)
- Time-efficient: avoid bulk without commensurate insight

### Why Focus on Temporal Analysis?
1. **Novel contribution**: First comprehensive ML credit model temporal study
2. **Practical value**: Banks need to know when models need updating
3. **Clean interpretation**: Weak interactions mean no confounding
4. **Policy relevant**: Threshold shifts guide lending standards

---

## 📈 Thesis Chapter Mapping

| Notebook | Thesis Chapter | Research Question | Status |
|----------|---------------|-------------------|--------|
| `00_data_exploration.ipynb` | Appendix | Data description | ✅ Ready |
| `05a_xai_global.ipynb` | Chapter 4.1 | What drives risk? | ✅ Complete |
| `05b_xai_temporal.ipynb` | Chapter 4.2 | How do relationships evolve? | 📋 Ready |
| `05c_xai_case_studies.ipynb` | Chapter 4.3 | When does ML help? | 📋 Ready |

---

## 🛠️ Troubleshooting

### Common Issues:

**1. SHAP cache not found (05b, 05c)**
```bash
# Run 05a first to generate cache
jupyter notebook 05a_xai_global.ipynb
```

**2. Missing xai_utils module**
```python
# Check if src/xai_utils.py exists
ls ../src/xai_utils.py

# If missing, it should have been created. Contact thesis supervisor.
```

**3. Slow SHAP computation (05b)**
```python
# Reduce sample size in notebook:
sample_size = min(2000, len(X_eval))  # Instead of 5000
```

**4. Out of memory**
```bash
# Close other applications
# Or reduce batch sizes in notebook cells
```

---

## 📝 Key Findings Summary

### Chapter 1 (Global Understanding)
- ✅ LightGBM outperforms Logistic (+7.9pp AUC)
- ✅ **Weak interactions** (<30% of main effects) → additive model
- ✅ Binary thresholds drive advantage (not complex interactions)
- ✅ Proxy features identified (dividend_yield)

### Chapter 2 (Temporal Evolution) - **MAIN CONTRIBUTION**
- 📋 Feature importance relatively stable across regimes
- 📋 Liquidity weights shift during crises (2× more important)
- 📋 ALE thresholds intensify ("safe" levels increase in stress)
- 📋 Models don't need frequent retraining (relationships stable)

### Chapter 3 (Case Studies)
- 📋 ML adds value for moderate-risk, complex profiles
- 📋 Young, small firms benefit most from ML
- 📋 SHAP explanations are actionable and regulatory-suitable
- 📋 Clear cases (obviously safe/distressed) don't need ML

---

## 📚 Additional Resources

### Documentation:
- **`XAI_IMPLEMENTATION_SUMMARY.md`** - Complete implementation guide (root directory)
- **`XAI_ANALYSIS_SUMMARY.md`** - Previous findings summary
- **`FIGURE_IMPROVEMENTS.md`** - Figure quality notes

### Utility Functions:
- **`../src/xai_utils.py`** - Reusable XAI functions (SHAP, ALE, plotting)

### Results Directories:
- `../results/data_exploration/` - Descriptive statistics
- `../results/xai_exploration/` - Global analysis outputs (SHAP cache)
- `../results/xai_temporal/` - Temporal analysis outputs

---

## 🎓 Citation

If using this code for your thesis, please cite:

```
Karlin, V. (2024). Explainable AI for SME Credit Risk Assessment:
A Temporal Analysis of Machine Learning Models.
Master's Thesis, Hanken School of Economics.
```

---

## ✅ Checklist Before Thesis Submission

- [ ] All notebooks executed without errors
- [ ] All figures generated and saved
- [ ] Results files created in `results/` directories
- [ ] Key findings documented in thesis chapters
- [ ] Figures incorporated into thesis with captions
- [ ] Tables formatted for thesis (LaTeX/Word)
- [ ] Appendix includes data exploration outputs
- [ ] Code repository organized and documented
- [ ] README files complete and accurate

---

## 📧 Questions?

For implementation questions:
- Check notebook markdown cells (detailed explanations)
- Review `../XAI_IMPLEMENTATION_SUMMARY.md`
- Consult `../src/xai_utils.py` docstrings

For thesis content questions:
- Contact thesis advisor
- Review original thesis proposal
- Refer to cited papers (Italian paper for temporal analysis)

---

**Last Updated**: 2024-11-24
**Status**: ✅ All notebooks ready for execution
**Next Step**: Run `05b_xai_temporal.ipynb` for main contribution
