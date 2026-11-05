# Diabetes Prediction Using Machine Learning

### End-to-End Data Science, Explainable AI & Machine Learning Application

**Python | Scikit-learn | XGBoost | SHAP | Streamlit | Plotly | MLOps Foundations**

---

## Project Overview

This repository presents an end-to-end Data Science and Machine Learning project developed to classify diabetes status using demographic, lifestyle, and clinical information.

The project demonstrates the complete machine learning development lifecycle, from data exploration, cleaning, preprocessing, and predictive modelling to model evaluation, Explainable Artificial Intelligence (XAI), and interactive application development.

Four supervised machine learning algorithms were investigated:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVC)
- XGBoost Classifier

In the original experiment, **XGBoost achieved the strongest reported performance, with 96.85% accuracy and 97.64% ROC-AUC**.

The project was subsequently extended with an improved machine learning workflow and an interactive Streamlit dashboard.

The enhanced system introduces cross-validation-based model selection, validation-based decision threshold optimisation, additional evaluation diagnostics, reproducible preprocessing, interactive predictions, and SHAP-based explanations.

The objective is not simply to develop a high-performing classification model, but to demonstrate how a machine learning experiment can be developed into a more reliable, explainable, maintainable, and deployable application.

> **Disclaimer:** This project is intended exclusively for educational, research, and portfolio purposes. It has not undergone clinical validation and must not be used as a medical diagnostic system or as a replacement for professional healthcare advice.

---

## Project Objectives

The primary objective is to develop and evaluate machine learning models capable of classifying diabetes status from demographic, lifestyle, and clinical measurements.

The project aims to:

1. Understand the structure and characteristics of the diabetes dataset.
2. Identify and resolve relevant data quality issues.
3. Perform Exploratory Data Analysis (EDA).
4. Build reproducible preprocessing pipelines.
5. Investigate and address target class imbalance.
6. Train and compare multiple classification algorithms.
7. Evaluate classification performance using appropriate metrics.
8. Investigate false-positive and false-negative predictions.
9. Improve model selection and decision threshold methodology.
10. Explain model predictions using SHAP.
11. Save trained machine learning pipelines for inference.
12. Develop an interactive prediction dashboard.
13. Introduce automated testing and reproducible project configuration.
14. Apply responsible artificial intelligence principles to healthcare-related modelling.

---

# Data Science Lifecycle

## 1. Business Understanding

Diabetes is a chronic medical condition that can lead to serious health complications.

Machine learning provides an opportunity to investigate patterns in patient characteristics and clinical measurements associated with diabetes.

### Problem Statement

Can supervised machine learning algorithms accurately classify diabetes status using demographic, lifestyle, and clinical data?

### Machine Learning Approach

- **Learning Type:** Supervised Learning
- **Problem Type:** Binary Classification
- **Target Variable:** Diabetes
- **Target Classes:** 0 = No Diabetes, 1 = Diabetes
- **Primary Objective:** Classification and evaluation of diabetes status

The inclusion of HbA1c and blood glucose measurements means the model is designed to classify diabetes status using available clinical information, rather than predict future diabetes onset.

A separate model excluding laboratory measurements could be investigated for preliminary risk-screening research.

---

## 2. Data Collection

The project uses a structured diabetes prediction dataset containing approximately **100,000 patient records**.

After removing duplicate records during the original experiment, 96,146 observations remained.

### Dataset Features

| Feature | Description |
|---|---|
| Gender | Patient gender |
| Age | Patient age |
| Hypertension | History of hypertension (0/1) |
| Heart Disease | History of heart disease (0/1) |
| Smoking History | Smoking behaviour category |
| BMI | Body Mass Index |
| HbA1c Level | Glycated haemoglobin measurement |
| Blood Glucose Level | Blood glucose measurement |
| Diabetes | Target classification variable |

### Dataset Characteristics

The dataset contains both numerical and categorical features.

The target variable is imbalanced, with substantially fewer diabetes-positive records than diabetes-negative records.

This makes evaluation metrics such as recall, precision, F1-score, and precision–recall analysis particularly important.

---

## 3. Data Cleaning

Data cleaning was performed to prepare the dataset for reliable machine learning experimentation.

### Cleaning Procedures

- Dataset structure inspection.
- Data type validation.
- Missing value analysis.
- Duplicate detection and removal.
- Target class inspection.
- Numerical feature validation.
- Categorical feature inspection.
- Data quality assessment.

### Duplicate Removal

Duplicate records were removed during the original analysis.

The cleaned dataset contained:

**96,146 records**

### Importance of Data Cleaning

Data cleaning helps reduce avoidable inconsistencies, supports reliable preprocessing, and improves the quality of the data provided to machine learning algorithms.

Potential duplicate patient records and repeated observations remain important considerations when assessing data independence and generalisation.

---

## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to understand the relationships, distributions, and patterns present in the dataset.

### Analyses Performed

- Diabetes class distribution.
- Numerical feature distributions.
- Categorical feature distributions.
- Correlation analysis.
- Outlier inspection.
- BMI distribution.
- Age distribution.
- HbA1c distribution.
- Blood glucose distribution.
- Diabetes prevalence.

### Visualisation Techniques

- Histograms.
- Boxplots.
- Correlation heatmaps.
- Count plots.
- Distribution plots.

### Key Findings

The dataset showed a significant imbalance between diabetic and non-diabetic observations.

HbA1c level and blood glucose level were among the strongest predictors associated with diabetes classification.

Other influential variables included:

- Age
- BMI
- Hypertension
- Heart disease

These findings informed subsequent modelling and explainability analysis.

---

## 5. Data Preprocessing

A reproducible preprocessing workflow was developed using Scikit-learn.

### Numerical Features

Numerical variables were processed using:

**Median Imputation**

Replaces missing numerical values with the median of the corresponding training feature.

**StandardScaler**

Standardises numerical features to improve compatibility with algorithms sensitive to feature scale.

### Categorical Features

Categorical variables were processed using:

**Most-Frequent Imputation**

Handles missing categorical values using the most frequently observed category in the training data.

**One-Hot Encoding**

Transforms categorical variables into numerical indicator features.

Unknown categories are handled using:

`OneHotEncoder(handle_unknown="ignore")`

### Pipeline Integration

The preprocessing workflow uses:

- ColumnTransformer
- Scikit-learn Pipeline
- Imbalanced-learn Pipeline

Integrating preprocessing with model training helps ensure that the same fitted transformations are applied during inference.

It also reduces the risk of data leakage when transformations are fitted correctly within training and cross-validation procedures.

---

## 6. Handling Class Imbalance

The original project used **SMOTE (Synthetic Minority Over-sampling Technique)** to address the underrepresentation of diabetes-positive observations.

SMOTE generates synthetic examples of the minority class.

### Original Approach

The original model used SMOTE within the machine learning pipeline.

However, applying standard SMOTE after one-hot encoding can generate fractional values for features representing categorical indicators.

### Improved Approach

The revised workflow uses **RandomOverSampler**, which balances training classes by resampling existing minority-class observations.

This avoids generating synthetic fractional categorical indicator values.

The resampling step is applied within the training pipeline rather than directly to the entire dataset.

### Why This Matters

Class imbalance can cause a classifier to favour the majority class.

Handling imbalance is intended to improve learning from underrepresented observations, although the effectiveness of each technique must be established through validation.

Future experiments may compare:

- Random oversampling.
- Class weighting.
- XGBoost's `scale_pos_weight`.
- Categorical-aware synthetic resampling.

---

# Machine Learning Model Development

## 7. Classification Algorithms

Four supervised machine learning algorithms were explored.

### Logistic Regression

A linear classification algorithm used as an interpretable baseline.

It estimates the probability of a binary outcome using a logistic function.

### Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees.

It can model nonlinear relationships and interactions between features.

### Support Vector Machine (SVC)

A classification algorithm that aims to identify a decision boundary between classes.

The original SVC experiment used a smaller training subset to manage computational requirements.

### XGBoost Classifier

An optimised gradient-boosted tree algorithm.

XGBoost builds successive trees to improve predictive performance by learning from errors made by earlier trees.

### Model Comparison

The algorithms were compared to investigate their predictive performance on the diabetes dataset.

The original experiment used similar preprocessing principles, although computational settings and training sample sizes were not identical for every algorithm.

The revised workflow focuses on more consistent development and evaluation of its primary candidate models.

---

## 8. Model Evaluation

Machine learning models were evaluated using several complementary performance metrics.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correct predictions |
| Precision | Measures how many predicted positive records are actually positive |
| Recall | Measures how many actual positive records are correctly identified |
| F1-score | Balances precision and recall |
| ROC-AUC | Evaluates discrimination across classification thresholds |
| Confusion Matrix | Displays true and false classifications |
| Cross-Validation | Estimates model performance across training-data partitions |
| Threshold Analysis | Investigates the relationship between classification thresholds and prediction outcomes |
| PR-AUC | Summarises precision–recall performance |
| Calibration Analysis | Investigates whether predicted probabilities reflect observed frequencies |

### Why Accuracy Alone Is Insufficient

When the dataset is imbalanced, a model may achieve high accuracy by predicting the majority class frequently.

For diabetes classification, recall is particularly relevant because false-negative predictions represent diabetes-positive records that the model fails to identify.

Precision is also important because excessive false positives may lead to unnecessary follow-up assessments.

Model evaluation therefore requires consideration of multiple metrics and their trade-offs.

---

## 9. Original Experiment — Best Performing Model

In the original experiment, **XGBoost was reported as the strongest-performing model**.

### Original XGBoost Results

| Metric | Score |
|---|---|
| Accuracy | 96.85% |
| Precision | 89.44% |
| Recall | 72.88% |
| F1-score | 80.31% |
| ROC-AUC | 97.64% |

### Results Interpretation

The original XGBoost model demonstrated strong overall predictive performance.

Its ROC-AUC score indicated strong discrimination between diabetes-positive and diabetes-negative records.

However, the recall score highlighted an important limitation.

The original test confusion matrix showed:

| Classification Outcome | Number |
|---|---:|
| True Positives | 1,236 |
| True Negatives | 17,388 |
| False Positives | 146 |
| False Negatives | 460 |

The model failed to identify approximately 27% of the diabetes-positive records at the evaluated classification threshold.

This observation motivated further investigation into decision threshold selection, model calibration, and recall-focused evaluation.

**Important:** These figures belong to the original experiment. They should not be interpreted as independently verified results from the improved workflow.

---

# Improved Machine Learning Workflow

## 10. Cross-Validation-Based Model Selection

The revised machine learning workflow improves the separation between model development and final testing.

### Original Limitation

The original workflow selected its strongest model using results from the test dataset.

Using test performance to select a model can introduce selection bias and produce overly optimistic estimates of generalisation performance.

### Improved Methodology

The enhanced workflow selects its preferred model using cross-validation performance on the development data.

The model-selection metric is mean cross-validation ROC-AUC.

The final holdout dataset is then reserved for evaluation of the selected model.

### Cross-Validation

The improved notebook uses three-fold stratified cross-validation in its current configuration.

Stratification helps preserve the approximate class distribution across validation folds.

### Benefits

- Reduces reliance on test-set results for model selection.
- Supports more transparent experimental comparisons.
- Improves reproducibility.
- Provides a more defensible evaluation methodology.

---

## 11. Decision Threshold Optimisation

Binary classifiers commonly use a probability threshold of 0.50 to convert predicted probabilities into class labels.

However, this threshold is not necessarily appropriate for every application.

### Improved Approach

The revised workflow investigates decision thresholds using validation predictions rather than selecting a threshold based on final test-set results.

A research recall target of **90%** is included in the improved configuration.

This value represents an experimental objective and does not imply that the model achieved 90% recall.

### Why Threshold Optimisation Matters

Reducing a classification threshold can increase recall, but may also increase false-positive predictions.

Threshold optimisation therefore requires balancing:

- Sensitivity.
- Specificity.
- Precision.
- False-positive rates.
- False-negative rates.

Any threshold intended for clinical use would require further validation and a clinically justified decision framework.

---

## 12. Additional Model Evaluation

The improved workflow introduces additional diagnostic analyses.

### Precision–Recall Analysis

Precision–recall evaluation is especially informative when the positive class is relatively uncommon.

### Probability Calibration

Calibration analysis investigates the relationship between predicted probabilities and observed outcomes.

Good discrimination does not necessarily imply that predicted probabilities are accurately calibrated.

### Subgroup Evaluation

Subgroup analysis investigates performance differences across available demographic groups, including age and gender.

These analyses provide an initial basis for examining potential performance disparities.

They do not establish comprehensive model fairness or clinical validity.

---

# Explainable Artificial Intelligence (XAI)

## 13. SHAP Explainability

SHAP (SHapley Additive Explanations) was used to investigate the behaviour of the trained machine learning model.

SHAP estimates how individual input features contribute to a prediction relative to a model baseline.

### Explainability Techniques

**Feature Importance**

Provides an overview of influential variables.

**SHAP Summary Plot**

Shows the distribution and direction of feature contributions across observations.

**SHAP Waterfall Plot**

Explains how individual features contribute to a particular prediction.

**Individual Prediction Explanations**

Provide model-specific insight into the factors influencing a selected output.

### Important Features Identified

The original analysis identified the following influential predictors:

1. HbA1c Level
2. Blood Glucose Level
3. Age
4. Hypertension
5. BMI
6. Heart Disease

### Importance of Explainability

Explainable AI supports transparency and helps developers investigate model behaviour.

However, SHAP explanations describe relationships learned by the model and do not establish medical causation.

---

# Interactive Diabetes Prediction Dashboard

## 14. Streamlit Dashboard

The project has been extended with an interactive machine learning dashboard developed using **Streamlit**.

The dashboard transforms the saved classification pipeline into a more accessible research demonstration.

Rather than requiring users to interact directly with Jupyter Notebook cells, the application provides an interface for entering example patient information and exploring predictions.

### Dashboard Sections

The application includes five principal sections:

- Prediction
- Explainability
- Performance
- Monitoring
- About

---

## 15. Interactive Prediction

The Prediction section allows users to enter patient features through an interactive interface.

### Input Features

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI
- HbA1c Level
- Blood Glucose Level

### Prediction Workflow

1. The user enters example patient information.
2. The application validates the input values.
3. The saved preprocessing pipeline transforms the input.
4. The trained classification model generates a probability.
5. The saved decision threshold determines the predicted class.
6. The application displays the model output.

The objective is to demonstrate reusable machine learning inference through a user-friendly interface.

Prediction results are research outputs and must not be interpreted as clinical diagnoses.

---

## 16. Dashboard Explainability

The Explainability section integrates SHAP-based explanations.

It is designed to help users investigate why the model generates a particular prediction.

### Features

- Individual prediction explanations.
- Feature contribution visualisations.
- Model interpretation.
- Identification of influential input variables.

Explainability helps improve transparency but does not guarantee that a prediction is accurate or clinically meaningful.

---

## 17. Model Performance Dashboard

The Performance section presents available evaluation information from the model's saved artifacts.

### Performance Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision–recall diagnostics
- Confusion matrix

Performance visualisations help users understand the strengths and weaknesses of the trained classifier.

The dashboard distinguishes historical results from the original experiment and metrics generated by the revised model.

---

## 18. Model Monitoring

The Monitoring section provides session-level information about prediction activity.

### Current Monitoring Capabilities

- Aggregate prediction activity.
- Current-session monitoring.
- Model information.
- Evaluation metadata.

The dashboard provides a foundation for future monitoring capabilities.

### Planned Monitoring Improvements

- Input data drift detection.
- Production data quality monitoring.
- Model performance monitoring when verified outcome labels are available.
- Model version comparison.
- Monitoring alerts.

The current dashboard should not be confused with a complete production monitoring system.

---

## 19. Model Information and Evaluation Provenance

The dashboard includes metadata documenting how the machine learning model was developed and evaluated.

### Available Metadata

| Field | Example Value |
|---|---|
| Selected Algorithm | XGBoost |
| Model Selection Metric | Cross-Validation ROC-AUC |
| Cross-Validation Folds | 3 |
| Cleaned Dataset Records | 96,146 |
| Training Records | 76,916 |
| Holdout Records | 19,230 |
| Validation Recall Target | 90% |

These values reflect the metadata observed during project development.

### Why Evaluation Provenance Matters

Evaluation provenance allows developers to understand the methodology behind a model's reported performance.

It helps document:

- How the model was selected.
- Which validation strategy was applied.
- How much data was used.
- Which experimental objectives were configured.

### Dashboard Presentation

The current application can display technical model metadata in JSON format.

A future interface improvement will present this information through a professional Model Information panel, while retaining raw JSON under an optional Developer Details section.

This will make the dashboard more accessible to nontechnical users without removing valuable technical information.

---

# Model Persistence and Application Development

## 20. Saving the Machine Learning Model

Joblib is used to save the trained preprocessing and classification pipeline.

Saving the pipeline enables inference without retraining the entire model for every new prediction.

### Benefits

- Reusable predictions.
- Consistent preprocessing.
- Integration with applications.
- Reduced inference setup requirements.
- Reproducible model loading.

The improved workflow also saves model metadata and information relating to the selected decision threshold.

### Deployment Status

The Streamlit dashboard has been implemented as a locally runnable application.

Its source code includes prediction, explainability, evaluation, and monitoring functionality.

Full end-to-end verification with the original dataset and trained model artifacts remains an important development milestone.

The application is not currently presented as clinically validated or publicly deployed.

---

## 21. Running the Application Locally

### Step 1 — Clone the Repository

```bash id="d8sg21"
git clone https://github.com/valeshiap14/diabetes-prediction-model.git
cd diabetes-prediction-model
```

### Step 2 — Install Dependencies

```bash id="eq7u91"
pip install -r requirements.txt
```

### Step 3 — Prepare the Dataset

Place the diabetes dataset at:

```text id="n2fk39"
data/diabetes_prediction_dataset.csv
```

### Step 4 — Generate the Model Artifacts

Run the improved training notebook to generate the trained pipeline and evaluation metadata.

### Step 5 — Launch the Dashboard

```bash id="j4bv68"
streamlit run app/main.py
```

Streamlit will launch the application in a browser.

**Note:** These instructions require the improved notebook, application files, dependencies, and project directory structure to be present in the repository. The dataset must be obtained from its authorised source if it is not included.

---

# Software Engineering and Testing

## 22. Automated Testing

The upgraded project includes automated tests using Pytest.

### Testing Areas

- Input validation.
- Numerical boundaries.
- Prediction-related data structures.
- Application helper functionality.

Automated testing reduces the likelihood of unnoticed regressions when application code is updated.

### Continuous Integration

GitHub Actions configuration has been added to support automated testing.

This provides a foundation for continuous integration and future CI/CD development.

### Future Engineering Improvements

- Expanded integration testing.
- End-to-end inference testing.
- Automated dependency checks.
- Docker-based deployment.
- Dedicated prediction APIs.
- Model version management.

---

# Technologies Used

## Programming and Development

- Python
- Jupyter Notebook
- Git
- GitHub

## Data Science and Numerical Computing

- Pandas
- NumPy
- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn

## Explainable AI

- SHAP

## Application Development

- Streamlit
- Plotly

## Model Persistence

- Joblib

## Software Quality

- Pytest
- GitHub Actions

## Supporting Python Libraries

- Pathlib
- Warnings

---

# Skills Demonstrated

## Data Science

- Business and problem understanding.
- Data cleaning.
- Data wrangling.
- Exploratory Data Analysis.
- Data visualisation.
- Statistical interpretation.
- Feature analysis.

## Machine Learning

- Supervised classification.
- Model training and comparison.
- Class imbalance handling.
- Preprocessing pipelines.
- Cross-validation.
- Model evaluation.
- Decision threshold analysis.
- Probability calibration diagnostics.

## Explainable AI

- SHAP integration.
- Feature importance.
- Individual prediction interpretation.
- Model transparency.

## Software Engineering

- Modular application development.
- Input validation.
- Model serialization.
- Reusable inference pipelines.
- Automated unit testing.
- Version control.

## MLOps Foundations

- Reproducible machine learning workflows.
- Model metadata.
- Evaluation provenance.
- Application integration.
- Continuous integration configuration.
- Monitoring concepts.

## Responsible AI

- Healthcare-related model limitations.
- Dataset bias awareness.
- Fairness considerations.
- Privacy awareness.
- Human oversight.
- Evaluation transparency.

---

# Project Limitations

Although the original experiment achieved strong reported classification performance, several limitations remain.

### Dataset Imbalance

The diabetes-positive class is substantially smaller than the diabetes-negative class.

### False-Negative Predictions

The original model missed approximately 27% of diabetes-positive test records at the evaluated classification threshold.

### External Validation

The model has not been independently validated on an external patient dataset.

### Dataset Generalisation

Results obtained on the development dataset may not generalise to other populations or healthcare environments.

### Clinical Feature Availability

HbA1c and blood glucose measurements are directly relevant to diabetes assessment.

Their inclusion limits the interpretation of the model as an early risk-screening tool that operates without laboratory testing.

### Model Calibration

Strong classification performance does not guarantee that predicted probabilities are accurately calibrated.

### Explainability Limitations

SHAP explains model behaviour but does not establish causal medical relationships.

### Clinical Validation

The application has not undergone prospective clinical evaluation or regulatory assessment.

It must not be used to make medical decisions.

---

# Ethical Considerations and Responsible AI

Healthcare-related artificial intelligence requires particular attention to safety, privacy, fairness, and transparency.

This project recognises the importance of:

- Protecting sensitive patient information.
- Avoiding unauthorised patient data collection.
- Understanding potential dataset bias.
- Investigating performance differences across demographic groups.
- Providing transparent explanations.
- Documenting model uncertainty and limitations.
- Maintaining human oversight.
- Avoiding unsupported clinical claims.

Synthetic demonstration records should be used when presenting the application publicly.

The model is designed as an educational research prototype and not as a replacement for healthcare professionals.

---

# Project Development Roadmap

## Completed

- Dataset exploration and cleaning.
- Exploratory Data Analysis.
- Initial model preprocessing.
- Comparison of four original classification algorithms.
- Original model evaluation and visualisation.
- SHAP explainability.
- Model persistence using Joblib.
- Revised cross-validation-based model selection.
- Validation-based threshold selection.
- Additional evaluation diagnostics.
- Streamlit dashboard implementation.
- Input validation and automated unit tests.
- GitHub Actions testing configuration.

## Integration and Validation

- Verify revised model performance using the original dataset.
- Complete end-to-end testing with the saved trained pipeline.
- Validate SHAP output for live application predictions.
- Synchronise the upgraded project with the GitHub repository.
- Update the technical report using confirmed evaluation results.
- Replace raw dashboard metadata with a more accessible information panel.

## Future Enhancements

- Hyperparameter optimisation using Optuna.
- FastAPI prediction service.
- Docker containerisation.
- MLflow experiment tracking.
- Model versioning and experiment management.
- Automated data drift detection.
- External dataset validation.
- Expanded fairness analysis.
- Cloud deployment using suitable hosting services.
- Production-oriented monitoring and security improvements.

---

# Project Outcomes

This project demonstrates the development of a supervised machine learning classification system and its extension into an interactive, explainable research application.

The original investigation showed how multiple algorithms could be trained and compared for diabetes classification.

The improved workflow addresses important methodological concerns involving model selection, validation, threshold selection, reproducibility, and evaluation transparency.

The Streamlit application demonstrates how a trained machine learning pipeline can be integrated into an interface for prediction, explainability, and model performance analysis.

Together, these components showcase practical skills relevant to:

**Data Scientist | Machine Learning Engineer | AI Engineer **

---

# Author

**Valeshia Pillay**

Bachelor of Science in Information Technology (BSc IT)

Aspiring Data Scientist | AI Engineer

GitHub: https://github.com/valeshiap14

---

# Acknowledgements

This project was developed as part of a Data Science, Machine Learning, and Artificial Intelligence portfolio.

Its objective is to demonstrate practical experience in healthcare-related analytics, predictive modelling, explainable AI, reproducible machine learning workflows, and application development.

The project continues to evolve through improved evaluation methods, software engineering practices, and MLOps-focused enhancements.
