# Customer Segmentation Analysis

An exploratory data analysis and customer segmentation project using **Python** and **K-Means clustering**. The notebook investigates customer demographics, household information, product spending, purchasing channels, and campaign responses, then explores whether customers can be grouped by similar recorded characteristics.

> **Important:** The current clustering results are exploratory, not validated customer personas. The notebook calculates a standardized dataset but passes the unscaled data to t-SNE and K-Means. Consequently, high-magnitude numeric fields and the customer ID can dominate the clusters. Review the limitations and recommendations below before interpreting or using the segments.

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
2. **Handle missing records.** The dataset contains 24 missing values in `Income`; dropping incomplete rows leaves 2,216 records.
3. **Prepare fields.** The notebook splits the customer registration date into day, month, and year, removes the original date and two constant-valued columns, and encodes object-valued columns.
4. **Explore the data** with category counts, campaign-response count plots, and a correlation heatmap.
5. **Create a scaled dataset** with `StandardScaler`, and use t-SNE to visualize the customers in two dimensions.
6. **Compare candidate cluster counts** using the K-Means elbow method and inertia.
7. **Fit a five-cluster K-Means model** and display its assigned labels on the t-SNE plot.

## Observed findings

These are descriptive observations from the supplied data and one run of the notebook; they are not validated customer profiles:

- **Education:** Graduation is the most common category (1,127 customers, about 50%), followed by PhD (486, about 22%) and Master (370, about 17%).
- **Marital status:** Married (864, about 39%) and Together (580, about 26%) are the largest categories.
- **Income data quality:** 24 records have missing income. The raw maximum income is 666,666, far above the raw mean of approximately 52,247, and should be investigated as a potential influential outlier.
- **Cluster size imbalance:** In one five-cluster run, the cluster sizes were 719, 717, 7, 1, and 772. The very small groups suggest that the current model isolates extreme records rather than yielding five similarly sized, stable groups.
- **Observed spending differences:** In that same run, one large cluster had higher average income and wine and meat spending than the other large clusters. These differences are exploratory and may be driven by the current preprocessing and feature scales.
- **Choosing *k*:** The elbow curve decreases as more clusters are added, but does not show a clearly decisive elbow. Five clusters should be treated as a provisional choice, not as a uniquely established answer.

The exact cluster labels, sizes, and profiles can vary with the installed scikit-learn version and model initialization defaults. Cluster numbers are arbitrary identifiers; a larger label does not mean a higher-value customer.

## What this project teaches

Working through this project highlights several useful lessons in data analysis and machine learning:

1. **Start with data quality.** Missing values and extreme or implausible records can materially influence later analysis. Identify and investigate them before drawing conclusions.
2. **Preprocessing shapes the model.** K-Means relies on distances. Features with larger numeric ranges can outweigh smaller-scale features unless the actual feature matrix used for clustering is scaled appropriately.
3. **Encoding is a modeling decision.** Assigning integer codes to nominal categories creates artificial order and distance. Select an encoding strategy that matches the feature meaning and the model.
4. **Identifiers are not behavioral features.** A customer ID generally identifies a record; it should not be used as evidence that two customers are behaviorally similar.
5. **Treat visualizations as evidence with limits.** t-SNE is a projection for exploration, not proof of natural segments. Count plots show counts, not necessarily within-category response rates.
6. **Cluster selection requires judgment.** Inertia always falls when more clusters are added. Use the elbow plot alongside other validation measures, stability checks, cluster sizes, and business interpretability.
7. **Unsupervised results need human interpretation.** Cluster labels do not explain why customers behave a certain way. Profile clusters with original, understandable fields and validate whether the groups are useful for a clearly defined business question.
8. **Communicate uncertainty.** A polished analysis should include limitations and next steps, not just plots or a model output.

## Limitations and recommended next steps

The current notebook is a useful learning exercise, but its five-cluster output is not yet suitable for customer targeting:

- The notebook creates a standardized matrix named `data`, but the t-SNE and K-Means cells use `df` instead. The model therefore uses **unscaled features**.
- The `ID` column remains in the feature matrix even though it is an identifier.
- `LabelEncoder` assigns integer values to nominal categories such as education and marital status, which can create misleading distances.
- The registration date is split into day, month, and year instead of being represented as a meaningful tenure feature.
- The extreme income value and unusually early birth year should be checked against documented data-quality rules.
- The five-cluster solution includes very small groups in an observed run. Cluster stability and usefulness have not been established.
- Campaign-response plots show counts; normalized response rates are needed for fairer comparisons between categories.

Recommended improvements:

1. Exclude identifiers and investigate unusual values before fitting the model.
2. Build an intentional feature matrix, encode categories appropriately, and scale the features actually supplied to K-Means and t-SNE.
3. Derive registration tenure using a documented reference date.
4. Reassess the cluster count with multiple diagnostics, such as silhouette scores, cluster-size review, and stability across runs.
5. Profile validated clusters on human-readable source columns and compare normalized campaign-response rates.

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

