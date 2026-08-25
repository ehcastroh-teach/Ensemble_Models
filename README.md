# Ensemble Models

A two-notebook series on exploratory data analysis, feature engineering, and ensemble learning. The notebooks use the Titanic passenger dataset as a running case study, progressing from raw data to a competitive ensemble of AutoML predictions.

## Learning Objectives

After working through both notebooks, you will be able to:

- Explore a real tabular dataset using summary statistics, distribution plots, and survival-rate comparisons
- Handle missing values with group-based median imputation
- Extract new features from raw columns using regular expressions and domain reasoning
- Encode categorical variables for machine learning using binary and ordinal representations
- Explain how ensemble methods reduce prediction error through model diversity
- Use LightAutoML and FlaML to build time-budgeted AutoML classification pipelines
- Combine predictions from multiple models using simple averaging and threshold tuning

## Data / File Dictionary

| File | Description |
|---|---|
| `train.csv` | Titanic training set - 891 passengers with survival labels |
| `test.csv` | Titanic test set - 418 passengers without labels (used for submission) |
| `gender_submission.csv` | Baseline Kaggle submission predicting survival by sex |
| `h2oai_experiment_macewube_test_predictions.csv` | Probability predictions from a separately-run H2O AutoML experiment |
| `01_titanic_eda_and_feature_engineering.ipynb` | Notebook 1 - EDA, feature engineering, and preprocessing |
| `02_ensemble_methods_and_automl.ipynb` | Notebook 2 - Ensemble concepts, LightAutoML, FlaML, and model averaging |
| `requirements.txt` | Python package dependencies |

## Workflow

```
train.csv + test.csv
        |
        v
01_titanic_eda_and_feature_engineering.ipynb
  - Part 1: Dataset overview and loading
  - Part 2: Exploratory data analysis
  - Part 3: Feature engineering (8 techniques)
  - Part 4: Final feature set and correlation analysis
        |
        v
02_ensemble_methods_and_automl.ipynb
  - Part 1: Ensemble learning concepts
  - Part 2: LightAutoML pipeline
  - Part 3: FlaML pipeline
  - Part 4: Combining predictions + threshold tuning
        |
        v
ensemble_avg_th050.csv (Kaggle submission)
```

## Step-by-Step Walkthrough

**Notebook 1** starts with the raw Titanic CSV files and builds a clean, numeric feature matrix. The key decisions along the way are:

- Titles are extracted from passenger names using a regex (`[A-Za-z]+\.`) because title encodes social status and age category simultaneously - two factors that directly influenced lifeboat priority. Rare titles are grouped into a single "Rare" category to avoid overfitting.
- Age is imputed using the median of each (sex, class) subgroup rather than the global median. A male third-class passenger has a systematically different age distribution than a female first-class passenger; ignoring that structure produces a worse imputation.
- `IsAlone` replaces `SibSp`, `Parch`, and `FamilySize` because the binary signal (alone vs. with any family) captures most of the survival variation with fewer columns and no ordinal assumptions.
- Fare is binned into quartiles because the raw distribution is right-skewed with extreme outliers. The quartile rank is robust to the skew and still preserves the ordering.
- Cabin is dropped (77% missing) and Ticket is dropped (no extractable structure).

**Notebook 2** explains ensemble theory first, then demonstrates it. The key ideas are:

- Ensemble methods reduce error when constituent models make *different* mistakes - model diversity is what makes averaging better than any individual model.
- LightAutoML and FlaML both search over multiple model families within a time budget, but use different search strategies. LightAutoML builds an internal blend; FlaML returns the single best configuration.
- Simple averaging of probability outputs from three AutoML systems (LightAutoML + FlaML + H2O.ai) achieves 78.2% Kaggle accuracy, beating each system run alone.
- Threshold tuning (shifting the decision boundary away from 0.5) changes the recall/precision balance. The correct place to tune the threshold is on a validation set, not the test set.

## How to Run

```bash
# 1. Clone the repo and navigate into it
# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and open the notebooks in order
jupyter notebook 01_titanic_eda_and_feature_engineering.ipynb
jupyter notebook 02_ensemble_methods_and_automl.ipynb
```

**Runtime note:** Notebook 1 completes in under 5 minutes on any modern laptop. Notebook 2 runs each AutoML framework for 300 seconds (5 minutes) - total runtime is approximately 15 minutes. Reduce the `TIMEOUT` constant to shorten training at the cost of search quality.

**Colab:** Both notebooks run on Google Colab with no local setup. Upload `train.csv`, `test.csv`, and `h2oai_experiment_macewube_test_predictions.csv` to your Colab session before running.

## Key Concepts Glossary

| Term | Definition |
|---|---|
| **Baseline accuracy** | The accuracy of a trivial classifier (e.g., always predict the majority class). Any model scoring below this is worse than doing nothing. |
| **Feature engineering** | Transforming raw attributes into representations that reveal patterns more useful to a learning algorithm. |
| **One-hot encoding** | Representing a categorical variable with $k$ levels as $k$ binary columns - one column active per row. |
| **Group-based imputation** | Filling missing values using the median (or mean) of a subgroup rather than the global statistic, to preserve conditional distributions. |
| **Ensemble method** | A learning strategy that combines predictions from multiple models to reduce total prediction error. |
| **Bagging** | Training multiple models on different bootstrap samples of the data and averaging their predictions. Reduces variance. |
| **Boosting** | Training models sequentially, each correcting the errors of the previous. Reduces bias. |
| **Out-of-fold (OOF) prediction** | A prediction made for a training row by a model that did not see that row during training - used to estimate generalization without touching the test set. |
| **AutoML** | Automated machine learning - frameworks that search over model families and hyperparameters within a time or evaluation budget. |
| **Decision threshold** | The probability cutoff above which a classifier predicts the positive class. Default is 0.5; tuning it adjusts the precision/recall tradeoff. |
| **AUC (Area Under the ROC Curve)** | A threshold-independent measure of classifier quality. 0.5 = random; 1.0 = perfect. |
| **Model diversity** | The degree to which ensemble members make different mistakes. Higher diversity produces larger ensemble gains. |

## Further Reading

- *The Elements of Statistical Learning* - Chapter 7 (bias-variance tradeoff) and Chapter 15 (Random Forests)
- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* - Chapter 7 (Ensemble Learning and Random Forests)
- Breiman (2001). Random Forests. *Machine Learning*, 45, 5-32.
- Dietterich (2000). Ensemble Methods in Machine Learning. *Multiple Classifier Systems*.

## Credits and Acknowledgements

- Titanic dataset: Kaggle Titanic Competition (https://www.kaggle.com/c/titanic)
- Titanic EDA and feature engineering approach: multiple public Kaggle kernels
- LightAutoML: sb-ai-lab/LightAutoML (https://github.com/sb-ai-lab/LightAutoML)
- FlaML: microsoft/FLAML (https://github.com/microsoft/FLAML)
- Ensemble theory foundations: Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*
---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
