# Customer Segmentation Analysis

An exploratory data analysis and customer segmentation project using **Python** and **K-Means clustering**. The notebook investigates customer demographics, household information, product spending, purchasing channels, and campaign responses, then explores whether customers can be grouped by similar recorded characteristics.

> **Important:** The clustering results are exploratory, not validated customer personas. The notebook now excludes the customer ID, imputes missing values, one-hot encodes nominal categories, robust-scales numeric fields, and uses the same prepared features for t-SNE and K-Means. The segments still need stability checks and business validation before they can support customer targeting.

## Project objectives

- Inspect and prepare a customer dataset for exploratory analysis.
- Explore its demographic categories and campaign-response patterns.
- Apply K-Means and visualize the assigned cluster labels.
- Practice evaluating the effect of feature preparation on distance-based models.
- Document findings, caveats, and next steps for a more reliable segmentation.

## Repository contents

```text
.
├── main.ipynb
├── README.md
└── data/
    └── customer-data.csv
```

## Dataset overview

The CSV contains **2,240 customer records and 29 columns**. Its fields cover:

- **Customer profile:** birth year, education, marital status, and income.
- **Household:** number of children and teenagers in the home.
- **Purchasing behavior:** spending across product categories, deal purchases, and web, catalog, and store purchases.
- **Engagement:** recency, monthly web visits, campaign acceptance indicators, and a response indicator.
- **Customer registration:** the date the customer was enrolled.

The data file is included in this project under `data/`. The notebook finds it using a relative path, so launch Jupyter from the project root.

## Analysis workflow

The analysis in [`main.ipynb`](./main.ipynb) follows these steps:

1. **Load and inspect the data** using the table dimensions, column types, summary statistics, missing-value checks, and unique-value counts.
2. **Check data quality.** The dataset contains 24 missing income values and two birth years before 1900. All 2,240 customer records are retained; missing numeric values are median-imputed in the modeling pipeline, and implausible birth years are flagged before deriving age.
3. **Prepare model features.** Exclude the customer ID and constant fields, derive tenure in days from the latest registration date in the dataset, one-hot encode nominal categories, and robust-scale numeric features.
4. **Explore the data** with category counts, within-category campaign-response rates, and a correlation heatmap of original numeric features.
5. **Create the model matrix** with a column-specific preprocessing pipeline and use that same matrix for t-SNE and K-Means.
6. **Compare candidate cluster counts** for *k* from 2 through 8 using inertia and silhouette score.
7. **Fit the selected K-Means model**, visualize its labels on the t-SNE projection, and profile clusters using original-scale fields and post-clustering campaign-response rates.

## Observed findings

These are descriptive observations from the supplied data and one run of the notebook; they are not validated customer profiles:

- **Education:** Graduation is the most common category (1,127 customers, about 50%), followed by PhD (486, about 22%) and Master (370, about 17%).
- **Marital status:** Married (864, about 39%) and Together (580, about 26%) are the largest categories.
- **Income data quality:** 24 records have missing income. The raw maximum income is 666,666, far above the raw mean of approximately 52,247, and should be investigated as a potential influential outlier.
- **Birth-year review:** Two records have birth years before 1900 (1893 and 1899). Their derived ages are marked missing and imputed for modeling; this handling is an explicit analysis assumption, not a confirmed correction to source data.
- **Cluster selection:** Among the tested values from 2 to 8, *k* = 2 produced the highest silhouette score in the reproducible run (approximately 0.331). The two clusters contained 1,506 (67.2%) and 734 (32.8%) customers.
- **Observed profile differences:** The larger cluster had average income of about 74,055, wine spending of 584, meat spending of 408, and a response rate of 23.8%. The other cluster had average income of about 41,600, wine spending of 168, meat spending of 49, and a response rate of 10.6%.
- **Interpretation:** Campaign-response fields were not used to create the clusters; response rates are compared afterward. The differences are descriptive associations and do not show that cluster membership causes campaign response.

Cluster labels are arbitrary identifiers; a larger label does not mean a higher-value customer. Small numeric differences can occur across dependency versions.

## What this project teaches

Working through this project highlights several useful lessons in data analysis and machine learning:

1. **Start with data quality.** Missing values and extreme or implausible records can materially influence later analysis. Identify and investigate them before drawing conclusions.
2. **Preprocessing shapes the model.** K-Means relies on distances. Features with larger numeric ranges can outweigh smaller-scale features unless the actual feature matrix used for clustering is scaled appropriately.
3. **Encoding is a modeling decision.** Assigning integer codes to nominal categories creates artificial order and distance. Select an encoding strategy that matches the feature meaning and the model.
4. **Identifiers are not behavioral features.** A customer ID generally identifies a record; it should not be used as evidence that two customers are behaviorally similar.
5. **Treat visualizations as evidence with limits.** t-SNE is a projection for exploration, not proof of natural segments. Response-rate plots improve comparisons between categories, but small groups still have less reliable estimates.
6. **Cluster selection requires judgment.** Inertia always falls when more clusters are added. Use the elbow plot alongside other validation measures, stability checks, cluster sizes, and business interpretability.
7. **Unsupervised results need human interpretation.** Cluster labels do not explain why customers behave a certain way. Profile clusters with original, understandable fields and validate whether the groups are useful for a clearly defined business question.
8. **Communicate uncertainty.** A polished analysis should include limitations and next steps, not just plots or a model output.

## Limitations and further validation

The notebook addresses the original preprocessing issues, but the result is still exploratory and **should not be used directly for customer targeting**:

- Robust scaling reduces the influence of extreme values but does not verify or correct the maximum income of 666,666. Confirm unusual values with a data dictionary or source owner.
- The rule treating birth years before 1900 as implausible is an assumption that should be reviewed with domain context.
- The highest silhouette score identifies *k* = 2 among the tested candidates, but silhouette alone does not establish that two clusters are stable or commercially useful.
- t-SNE is a visualization technique; visual separation in its projection is not independent evidence that the segments are reliable.
- Response-rate differences are observational and may reflect other customer or campaign factors. Small subgroups can also produce less reliable rates.

Recommended next steps are to test cluster stability across resampled data and initializations, review the data-quality rules, compare the segments against a specific marketing use case, and validate any proposed actions before operational use.

## Run the notebook

You need Python, Jupyter Notebook, and the packages used in the notebook: `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

From the project root, install the packages if needed and start Jupyter:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
python -m jupyter notebook
```

Open `main.ipynb` and run the cells from top to bottom. Running from the project root allows the notebook to locate `data/customer-data.csv`.

## Tools and methods

- **Python** for data analysis
- **pandas** and **NumPy** for data preparation
- **Matplotlib** and **Seaborn** for visualization
- **scikit-learn** for encoding, standardization, t-SNE, and K-Means
