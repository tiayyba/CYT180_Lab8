# CYT180 — Lab 8: Training and Evaluating Machine Learning Models

**Weight:** 3%
**Work Type:** Individual
**Submission Format:** Your Colab notebook printed to PDF (see Submission Guidelines)

---

## Introduction

In Lab 7, you prepared a network security dataset for machine learning. You checked its structure, removed duplicate records, filled missing values with the median, found outliers and decided to keep them (because many of them are the attacks), looked at correlations, saw that only about 10% of the connections are attacks, separated the features (X) from the target (y), and scaled the features.

In this lab, you continue with the **same dataset and the same sample**. You will move to the next stage of the machine learning workflow: you will split the data into a training set and a test set, scale the features the correct way, train a **logistic regression** model, and evaluate it with the metrics from the Week 10 lecture: the confusion matrix, accuracy, precision, recall, F1-score, the ROC curve, and AUC.

A model is only useful if it works on data it has never seen. So in this lab you will always train on one part of the data and test on another part. You will also see that one model can behave very differently depending on the **threshold** you use, and you will finish by choosing the setup that best fits the needs of a security team.

You will mostly run provided code and observe the results, and you will complete one small task in most sections.

**Every student works with their own sample of the data**, chosen by your student ID, exactly as in Lab 7. Your numbers will be different from your classmates' numbers. Submissions with another student's numbers will receive 0.

---

## Learning Objectives

By the end of this lab, you will be able to:

- Split a dataset into training and test sets with stratification
- Scale features without leaking information from the test set
- Train a logistic regression model and read its coefficients
- Generate predicted labels and predicted probabilities
- Build a confusion matrix and calculate accuracy, precision, recall, and F1-score
- Plot an ROC curve and interpret the AUC
- Explain how changing the threshold, or giving the attack class more weight, changes the results
- Choose a model setup for a security team and justify it with your own numbers

---

## About the Data

You use the same file as in Lab 7, `network_connections.csv`. Each row is **one network connection** to a company server. About 10% of the connections are **attacks**: port scans, brute-force logins, and data exfiltration.

| Column | Meaning |
|---|---|
| `conn_id` | an ID number for the connection |
| `duration_sec` | how long the connection lasted, in seconds |
| `bytes_sent` | bytes sent from the inside host to the outside |
| `bytes_received` | bytes received by the inside host |
| `packets` | total packets in the connection |
| `failed_logins` | failed login attempts during the connection |
| `conn_per_min` | how many connections the same source made in that minute |
| `unique_ports` | how many different ports the same source contacted in that minute |
| `hour_of_day` | hour the connection started (0 to 23) |
| `label` | the target: **0 = normal**, **1 = attack** |

In this lab, **positive** means **attack** (label 1) and **negative** means **normal** (label 0).

---

## Prerequisites

- You will work in **Google Colab**. Pandas, NumPy, scikit-learn, seaborn, and matplotlib are already installed there.
- Download `network_connections.csv` from Blackboard (the same file as Lab 7).
- In Colab, click the **folder icon** on the left, then the **upload** button, and choose the file. It will be saved as `/content/network_connections.csv`.
- Start a **new notebook** for this lab. You do not need your Lab 7 notebook: Section 1 rebuilds your Lab 7 data for you.

**Note:** files uploaded to Colab are deleted when the runtime restarts. If you get a "file not found" error, upload the file again.

---

## How the Tasks Work

Each task cell ends with an **answer line** that starts with `print("Task N |", STUDENT_ID, "|", ...)`. The line is already written for you, and it prints the variables that your code must create. Your job is to write the code above it that calculates those variables.

- **Do not change the answer line.** Your marks are read from it.
- If the answer line gives an error such as `NameError: name 'n_train' is not defined`, it means you have not created that variable yet.
- Round numbers exactly as the task says.

---

## Grading (10 marks)

| Item | Marks |
|---|---|
| Task 1 – Check your split | 1 |
| Task 2 – Check the scaling | 1 |
| Task 3 – Read the model's coefficients | 1 |
| Task 4 – Confusion matrix and metrics | 1 |
| Task 5 – AUC and overlapping probabilities | 1 |
| Task 6 – A lower threshold | 1 |
| Reflection questions R1–R4 (1 mark each) | 4 |
| **Total** | **10** |

The provided code cells are not graded on their own, but your notebook must show that you ran them. A task whose answer line is missing receives 0. A submission without the Section 0 cell, or with numbers that do not match your student ID, receives 0.

---

## Section 0 — Your Name, Student ID, and Your Sample

Run this cell **first**, at the top of your notebook. Type your own full name and student ID between the quotes before running it. Use the same student ID as in Lab 7.

```python
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score)

STUDENT_NAME = "Your Full Name"    # <-- type your full name
STUDENT_ID   = "000000000"         # <-- type your 9-digit student ID

if STUDENT_ID == "000000000" or STUDENT_NAME == "Your Full Name":
    raise ValueError("Type your own name and student ID in this cell first.")

# The last digit of your student ID picks your column, the same one as in Lab 7
MY_COLUMNS = ["duration_sec", "bytes_sent", "bytes_received", "conn_per_min", "unique_ports"]
MY_COLUMN = MY_COLUMNS[int(STUDENT_ID.strip()[-1]) % 5]

print("Name:      ", STUDENT_NAME)
print("Student ID:", STUDENT_ID)
print("My column: ", MY_COLUMN)
print("Timestamp: ", datetime.datetime.now().isoformat())
```

**Explanation**

- The first lines import the libraries. The new ones come from **scikit-learn**: `train_test_split` splits the data, `StandardScaler` scales it, `LogisticRegression` is the model, and the functions in `sklearn.metrics` calculate the evaluation metrics.
- `MY_COLUMN` is the same column you used for outliers in Lab 7. You will use it again in Task 2.

Use the variables `STUDENT_ID` and `MY_COLUMN` in later cells exactly as shown. Do not change them.

---

## Section 1 — Rebuild Your Lab 7 Data

This cell repeats the Lab 7 steps that changed the data, in one place. It takes your own 5,000-row sample, removes duplicate rows, fills the missing values in `duration_sec` and `bytes_received` with the median, and separates X and y. As we decided in Lab 7, we **keep** the outliers, because many of them are attacks.

We do **not** repeat the Lab 7 scaling here. You will scale again in Section 3, the correct way for model training.

```python
df_all = pd.read_csv("/content/network_connections.csv")

# Same sample as Lab 7: your student ID decides which rows you get
df = df_all.sample(n=5000, random_state=int(STUDENT_ID)).reset_index(drop=True)

# Lab 7, Section 3: remove duplicates and fill missing values with the median
df_clean = df.drop_duplicates().reset_index(drop=True)
for col in ["duration_sec", "bytes_received"]:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Lab 7, Section 8: separate features (X) and target (y)
X = df_clean.drop(columns=["conn_id", "label"])
y = df_clean["label"]

print("Rows in df_clean:     ", len(df_clean))
print("Missing values left:  ", df_clean.isnull().sum().sum())
print("Attacks (label = 1):  ", y.sum())
print("Feature columns:      ", list(X.columns))
```

**Explanation**

- The number of rows and attacks should be the same as in your Lab 7 notebook (Tasks 1, 3, and 5). If they are not, check that you typed the same student ID.
- `y.sum()` counts the attacks, because `label` is 1 for an attack and 0 for a normal connection.

---

## Section 2 — Split the Data into Training and Test Sets

A model must be tested on connections it has never seen, just like real traffic tomorrow. So we hold back part of the data as a **test set** and train only on the rest, the **training set**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,     # 25% of the rows go to the test set
    stratify=y,         # keep the same percentage of attacks in both sets
    random_state=42     # the same split every time you run the cell
)

print("X_train:", X_train.shape, "  X_test:", X_test.shape)
```

**Explanation**

- `train_test_split()` shuffles the rows and returns four parts: the features and labels of the training set (`X_train`, `y_train`) and of the test set (`X_test`, `y_test`).
- `test_size=0.25` puts 25% of the rows in the test set and 75% in the training set.
- `stratify=y` matters for imbalanced data. Without it, a random split could, by chance, put too few attacks in the test set, and the evaluation would not be reliable. With it, both sets keep the same percentage of attacks as the full data.
- `random_state=42` makes the shuffle repeatable. Your rows are still different from your classmates' rows, because your sample in Section 1 is different.

### Task 1 — Check Your Split (1 mark)

Create these four variables:

1. `n_train`: the number of rows in `X_train`
2. `n_test`: the number of rows in `X_test`
3. `pct_train`: the percentage of attacks in `y_train`, rounded to 1 decimal
4. `pct_test`: the percentage of attacks in `y_test`, rounded to 1 decimal

Hint: `len(X_train)` gives the number of rows. Because `label` is 0 or 1, `y_train.mean()` is the fraction of attacks; multiply it by 100 to get a percentage.

```python
# TODO: Task 1 - create n_train, n_test, pct_train, pct_test
# Your code here

print("Task 1 |", STUDENT_ID, "|", n_train, n_test, pct_train, pct_test)
```

The two percentages should be almost the same. That is what `stratify=y` does.

---

## Section 3 — Scale the Features (Fit on the Training Set Only)

In Lab 7, you fitted the scalers on all rows to see how scaling works. For a real model, that is a mistake: the scaler would learn the mean and standard deviation of the **test** rows too, so information from the test set would leak into training. This is called **data leakage**, and it makes the test results look better than they really are.

The correct order is:

```text
Split  →  fit the scaler on X_train  →  transform X_train and X_test with that same scaler
```

We use **standardization** (`StandardScaler`) instead of Min-Max normalization. In Lab 7 (R4) you saw that Min-Max squeezes almost all values of a column close to 0 when the column has extreme values. Standardization is less affected by this.

```python
scaler = StandardScaler()

# fit on the training set only, then transform it
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

# transform the test set with the SAME scaler (no fitting)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

print(X_train_scaled.describe().round(3))
```

**Explanation**

- `fit_transform(X_train)` learns the mean and standard deviation of each column **from the training rows only**, then scales them.
- `transform(X_test)` uses those same training values to scale the test rows. It does not learn anything from the test set.
- Look at the `mean` and `std` rows of the output: every training column now has mean 0 and standard deviation 1.

### Task 2 — Check the Scaling (1 mark)

For **your column** (`MY_COLUMN`), create:

1. `train_mean`: the mean of `X_train_scaled[MY_COLUMN]`, rounded to 3 decimals
2. `test_mean`: the mean of `X_test_scaled[MY_COLUMN]`, rounded to 3 decimals

```python
# TODO: Task 2 - create train_mean and test_mean
# Your code here

print("Task 2 |", STUDENT_ID, "|", MY_COLUMN, train_mean, test_mean)
```

**R1 (1 mark):** Your `train_mean` is 0 (it may print as 0.0 or -0.0), but your `test_mean` is not exactly 0. Using your own numbers, explain why the test mean is not 0, and why this is the correct result rather than a mistake. What would be wrong if we fitted the scaler on all rows before splitting? (2–3 sentences)

---

## Section 4 — Train a Logistic Regression Model

Logistic regression is the classifier from the Week 9 lecture. It gives each feature a **weight** (also called a **coefficient**), adds up the weighted features into a score, and turns the score into a probability between 0 and 1 with the sigmoid function. It is a good **baseline** model: it is fast, it gives probabilities, and its weights are easy to read.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# One coefficient per feature, sorted from largest to smallest
coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print(coefficients.round(3))
```

**Explanation**

- `fit()` is the training step. The model looks at the training rows and their labels and finds the weights that separate attacks from normal connections best.
- `max_iter=1000` gives the training algorithm enough steps to finish.
- `model.coef_[0]` holds the learned weights, one per feature. We put them in a Pandas Series with the column names so they are easy to read.
- Because all features were standardized, we can compare the weights. A **large positive** weight means a higher value of that feature pushes the prediction towards **attack**. A **negative** weight pushes it towards **normal**.

### Task 3 — Read the Model's Coefficients (1 mark)

Using the `coefficients` Series, create:

1. `top_feature`: the name of the feature with the **largest** coefficient
2. `top_coef`: its coefficient, rounded to 3 decimals
3. `low_feature`: the name of the feature with the **smallest** (most negative) coefficient
4. `low_coef`: its coefficient, rounded to 3 decimals

Hint: `coefficients.idxmax()` returns the name of the largest value, and `coefficients.max()` returns the value. `idxmin()` and `min()` work the same way.

```python
# TODO: Task 3 - create top_feature, top_coef, low_feature, low_coef
# Your code here

print("Task 3 |", STUDENT_ID, "|", top_feature, top_coef, low_feature, low_coef)
```

Think about whether your `top_feature` makes sense for the three kinds of attacks in this dataset.

---

## Section 5 — Make Predictions on the Test Set

Now the model sees the test set for the first time. We ask it for two kinds of output:

```python
# Predicted labels: 0 (normal) or 1 (attack), using the default threshold of 0.5
y_pred = model.predict(X_test_scaled)

# Predicted probability of attack (class 1) for each test connection
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Show the first 10 test connections
preview = pd.DataFrame({
    "actual": y_test.values[:10],
    "prob_attack": y_prob[:10].round(3),
    "predicted": y_pred[:10]
})
print(preview)
```

**Explanation**

- `predict()` returns a label for each connection. A connection is labelled **attack** when its probability is 0.5 or more. This cut-off is the **threshold**.
- `predict_proba()` returns two probabilities for each connection: the probability of class 0 and the probability of class 1. They add up to 1. `[:, 1]` keeps only the second one, the probability of **attack**.
- In the preview, compare `actual` with `predicted`. Most rows should match. The probability column shows how sure the model is.

---

## Section 6 — Confusion Matrix and Evaluation Metrics

Remember the four outcomes from the Week 10 lecture:

| | Predicted normal (0) | Predicted attack (1) |
|---|---|---|
| **Actually normal (0)** | True Negative (TN) | False Positive (FP): a false alarm |
| **Actually attack (1)** | False Negative (FN): a missed attack | True Positive (TP) |

And the four metrics:

- **Accuracy** = (TP + TN) / all rows: the fraction of all predictions that were correct
- **Precision** = TP / (TP + FP): of the connections the model flagged as attacks, the fraction that really were attacks
- **Recall** = TP / (TP + FN): of the real attacks, the fraction the model caught
- **F1-score**: one number that combines precision and recall

### Task 4 — Confusion Matrix and Metrics (1 mark)

Using `y_test` and `y_pred`, create:

1. `tn, fp, fn, tp`: the four counts of the confusion matrix
2. `acc, prec, rec, f1`: accuracy, precision, recall, and F1-score, each rounded to 3 decimals

Hint: `confusion_matrix(y_test, y_pred).ravel()` returns the four counts in the order TN, FP, FN, TP, so you can write `tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()`. For the metrics, use `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`. Each one takes `(y_test, y_pred)`.

```python
# TODO: Task 4 - create tn, fp, fn, tp and acc, prec, rec, f1
# Your code here

print("Task 4 |", STUDENT_ID, "|", tn, fp, fn, tp, "|", acc, prec, rec, f1)
```

**R2 (1 mark):** In Lab 7 (R3), you saw that a model that predicts "normal" for every connection already gets a high accuracy. Using your Task 1 numbers, what accuracy would that model get on your test set? Compare it with your model's accuracy. Then, using your own FP and FN counts, explain what each one would mean for a security team (SOC) in real life. (3–4 sentences)

---

## Section 7 — Visualize the Results: Confusion Matrix and ROC Curve

Numbers are easier to understand, and to explain to others, when you can see them. Run this cell to draw the confusion matrix as a heatmap and the ROC curve.

```python
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

# 1) Confusion matrix as a heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[0],
            xticklabels=["Normal (0)", "Attack (1)"], yticklabels=["Normal (0)", "Attack (1)"])
axs[0].set_xlabel("Predicted label")
axs[0].set_ylabel("Actual label")
axs[0].set_title("Confusion Matrix (threshold 0.5)")

# 2) ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axs[1].plot(fpr, tpr, label="Logistic regression")
axs[1].plot([0, 1], [0, 1], "k--", label="Random guessing")
axs[1].set_xlabel("False Positive Rate")
axs[1].set_ylabel("True Positive Rate (Recall)")
axs[1].set_title("ROC Curve")
axs[1].legend()

plt.tight_layout()
plt.show()
```

**Explanation**

- The heatmap shows the same four counts as Task 4. The numbers on the diagonal (top left and bottom right) are the correct predictions.
- `roc_curve()` tries every possible threshold. For each threshold, it calculates the **True Positive Rate** (recall) and the **False Positive Rate** (the fraction of normal connections that were flagged), and the plot connects these points.
- The dashed line is a model that guesses at random. The closer the curve gets to the **top-left corner**, the better the model separates attacks from normal traffic.
- The **AUC** (area under the ROC curve) sums the whole curve up in one number: 1.0 is perfect separation, and 0.5 is random guessing.

### Task 5 — AUC and Overlapping Probabilities (1 mark)

Create:

1. `auc_value`: the AUC of the model, rounded to 3 decimals. Use `roc_auc_score(y_test, y_prob)`.
2. `min_attack_prob`: the **lowest** attack probability that the model gave to a **real attack** in the test set, rounded to 3 decimals
3. `max_normal_prob`: the **highest** attack probability that the model gave to a **real normal** connection in the test set, rounded to 3 decimals

Hint: `y_prob[y_test.values == 1]` keeps only the probabilities of the real attacks. Then use `.min()`. Do the same with `== 0` and `.max()` for the normal connections.

```python
# TODO: Task 5 - create auc_value, min_attack_prob, max_normal_prob
# Your code here

print("Task 5 |", STUDENT_ID, "|", auc_value, min_attack_prob, max_normal_prob)
```

If `min_attack_prob` is lower than `max_normal_prob`, the probabilities of the two classes **overlap**: some attacks look "more normal" than some normal connections. No threshold can then be perfect, and choosing a threshold becomes a trade-off between missed attacks and false alarms.

---

## Section 8 — What Happens with a Lower Threshold?

By default, a connection is flagged as an attack when its probability is **0.5** or more. A security team can choose a different threshold. A **lower** threshold flags more connections, so it should catch more attacks, but it may also raise more false alarms.

You already have the probabilities in `y_prob`, so you do not need to train the model again. You only need to compare them with a new threshold:

```python
y_pred_03 = (y_prob >= 0.3).astype(int)
```

This line gives 1 (attack) to every connection with a probability of 0.3 or more, and 0 (normal) to the rest.

### Task 6 — A Lower Threshold (1 mark)

Using `y_test` and `y_pred_03`, create:

1. `tn3, fp3, fn3, tp3`: the four counts of the confusion matrix at threshold 0.3
2. `prec3, rec3`: precision and recall at threshold 0.3, each rounded to 3 decimals

```python
y_pred_03 = (y_prob >= 0.3).astype(int)

# TODO: Task 6 - create tn3, fp3, fn3, tp3 and prec3, rec3
# Your code here

print("Task 6 |", STUDENT_ID, "|", tn3, fp3, fn3, tp3, "|", prec3, rec3)
```

**R3 (1 mark):** Compare your results at threshold 0.5 (Task 4) and 0.3 (Task 6). How many more attacks did the lower threshold catch, and how many more false alarms did it raise? Explain why lowering the threshold moves recall and precision in these directions. (2–3 sentences, using your own numbers)

---

## Section 9 — Giving the Attack Class More Weight

At the end of Lab 7, we said that we would deal with the class imbalance in Lab 8. Only about 10% of the training rows are attacks, so during training the model "sees" far more normal connections and pays more attention to getting them right.

One simple fix is to give each **attack** row more weight during training. With `class_weight="balanced"`, scikit-learn makes the weights inversely proportional to the class sizes: a mistake on a rare attack row costs about 9 times more than a mistake on a normal row. Unlike copying attack rows (oversampling), this does not add any rows, so it cannot cause data leakage.

```python
model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
model_bal.fit(X_train_scaled, y_train)

y_pred_bal = model_bal.predict(X_test_scaled)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_bal).ravel()

print("Balanced model (threshold 0.5)")
print("TN, FP, FN, TP:", tn_b, fp_b, fn_b, tp_b)
print("Precision:", round(precision_score(y_test, y_pred_bal), 3))
print("Recall:   ", round(recall_score(y_test, y_pred_bal), 3))
```

**Explanation**

- The only change from Section 4 is `class_weight="balanced"`. The data, the split, and the scaling are the same, so the comparison is fair.
- Compare the four counts with your results from Task 4. Giving attacks more weight usually raises recall (fewer missed attacks) and lowers precision (more false alarms), just like a lower threshold does.

Now you have three setups to compare:

| Setup | Results from |
|---|---|
| A: Standard model, threshold 0.5 | Task 4 |
| B: Standard model, threshold 0.3 | Task 6 |
| C: Balanced model, threshold 0.5 | Section 9 |

**R4 (1 mark):** A company's SOC has two analysts. They can investigate every alert the model raises, but they cannot investigate thousands of connections by hand. Missing a data exfiltration would be very costly for the company. Which setup (A, B, or C) would you recommend for this SOC? Justify your choice with your own FP and FN numbers from the three setups, and explain the trade-off you are accepting. (3–4 sentences)

---

## Conclusion

In this lab, you continued from your Lab 7 data and completed the modelling and evaluation stage of the machine learning workflow. You split the data with stratification, scaled the features without leaking information from the test set, trained a logistic regression model, and read its coefficients. You then evaluated the model with a confusion matrix, accuracy, precision, recall, F1-score, the ROC curve, and AUC, and you saw that the threshold and the class weights change the balance between missed attacks and false alarms.

The most important lesson is that there is no single "best" result. The right setup depends on what the security team needs: how costly a missed attack is, and how many false alarms the analysts can handle.

---

## Submission Guidelines

1. **Run the whole notebook from top to bottom** (Runtime → Run all) so every output is visible. The Section 0 cell must be the first cell.
2. **Write the reflection answers (R1–R4) in a text cell** directly under the task they belong to. Start each answer with its label (for example, **R1:**). Answer in your own words and refer to **your own numbers**. Delete any template or AI-generated text.
3. **Print the notebook to PDF:** File → Print → Save as PDF. Check that the outputs and charts are readable and not cut off.
4. **Submit the PDF** on Blackboard, named `Lab8_FirstName_LastName.pdf`.

Deductions:

- A missing or changed answer line, or a missing reflection answer, receives 0 for that item.
- Missing Section 0 cell, or numbers that do not match your student ID: 0 for the lab.
