# Bitcoin Transaction Anomaly Detection with Graph Machine Learning

**A comparative study of feature-based anomaly detection, self-supervised graph embeddings, and graph convolutional networks on the Elliptic Bitcoin dataset.**

This project explores how different learning paradigms identify potentially illicit Bitcoin transactions under class imbalance. It evaluates classical distance- and density-based detectors, reconstruction-based autoencoders, a supervised classifier, a link-prediction-based embedding approach, and a graph-aware **Graph Convolutional Network (GCN)** within a common node-scoring framework.

The emphasis is on **understanding and comparing methodological choices**, rather than claiming a production-ready financial crime detection system or maximizing a single benchmark score.

> **At a glance:** 46,564 labeled transactions · 36,624 retained transaction links · 166 input columns · 7 evaluated methods · chronological train/validation/test splits · ROC-AUC, Average Precision, Precision, Recall, and F1.

## Contents

1. [Research question and objectives](#research-question-and-objectives)
2. [Dataset and graph construction](#dataset-and-graph-construction)
3. [Methodology](#methodology)
4. [Evaluation protocol](#evaluation-protocol)
5. [Experimental results](#experimental-results)
6. [Interpretation](#interpretation)
7. [Running the notebook](#running-the-notebook)
8. [Reproducibility and limitations](#reproducibility-and-limitations)
9. [Suggested extensions](#suggested-extensions)
10. [Data attribution and author](#data-attribution-and-author)

## Research question and objectives

**Can graph-aware learning exploit transaction connectivity to detect anomalous Bitcoin activity more effectively than models trained only on transaction-level features?**

The notebook approaches this question by:

- Generating a **continuous anomaly score for each labeled transaction**, where larger values indicate greater suspicion.
- Comparing **unsupervised, one-class, supervised, and self-supervised** approaches in a common workflow.
- Separating **ranking quality** (ROC-AUC and Average Precision) from **threshold-dependent decisions** (Precision, Recall, and F1).
- Examining the difference between models that treat nodes as independent feature vectors and a **GCN that propagates information over transaction edges**.
- Using chronologically ordered time-step splits and choosing classification thresholds from validation labels rather than test labels.

The comparison is primarily educational and exploratory. Not every baseline is expected to suit this dataset equally well.

## Dataset and graph construction

The notebook uses the public [Elliptic Bitcoin Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), consisting of transaction features, directed transaction links, and transaction-level labels.

### Original files

| File | Purpose |
| --- | --- |
| `elliptic_txs_features.csv` | Transaction identifiers, time-step information, and numerical features |
| `elliptic_txs_edgelist.csv` | Directed links between transactions (`txId1`, `txId2`) |
| `elliptic_txs_classes.csv` | Transaction labels: illicit (`1`), licit (`2`), or `unknown` |

### Dataset sizes

| Property | Original data | Analyzed subset |
| --- | ---: | ---: |
| Transactions / nodes | 203,769 | **46,564** |
| Transaction links / edges | 234,355 | **36,624** |
| Input columns used as features | — | **166** |
| Distinct time steps | — | **49** |
| Licit (`0`) | — | **42,019 (90.24%)** |
| Illicit (`1`) | — | **4,545 (9.76%)** |

**Filtering matters.** The notebook excludes transactions with unknown class labels and keeps an edge only if *both* endpoint transactions remain in the labeled subset. As a result, the modeled graph is much smaller and sparser than the original Elliptic transaction graph. The 166 input columns include the time-step column; they should not all be described as independently engineered transaction attributes.

The notebook maps the labels to `0 = licit / normal` and `1 = illicit / anomaly`, creates a transaction-ID-to-row-index mapping, builds a NumPy `edge_index` array, and checks degree statistics. On the retained graph, the approximate mean undirected degree is **1.57**, while the observed maximum degree is **453**. For the GCN, original directed links are symmetrized and self-loops are added.

### Chronological split

Distinct time steps are ordered, then allocated approximately **60% / 20% / 20% by time step** to training, validation, and testing. This does *not* imply an exactly 60/20/20 split by number of transactions.

| Partition | Transactions | Illicit | Illicit prevalence |
| --- | ---: | ---: | ---: |
| Train | 26,381 | 2,871 | 10.88% |
| Validation | 8,999 | 1,038 | 11.53% |
| Test | 11,184 | 636 | 5.69% |

The lower positive-class prevalence in the test period is important when interpreting Precision, F1, and Average Precision.

## Methodology

The pipeline follows the sequence below:

```text
Elliptic CSV files
       │
       ▼
Merge labels → retain labeled transactions → filter graph edges
       │
       ▼
Chronological train / validation / test split
       │
       ├── Feature-only methods: kNN, LOF, AE, One-Class AE, Logistic Regression
       │
       ├── Self-supervised node-ID embeddings: link prediction → LOF
       │
       └── Graph-aware supervised model: two-layer GCN
       │
       ▼
Node anomaly scores → validation-selected threshold → test-set metrics
```

### 1. Distance- and density-based baselines

**kNN distance.** Each transaction receives a score equal to the average distance to its nearest neighbors in feature space (the implementation requests `k=10` neighbors and discards the self-match). This approach detects geometric isolation but does not use transaction edges.

**Local Outlier Factor (LOF).** LOF compares the local density around a transaction with densities around neighboring observations (`n_neighbors=20`, `novelty=True`). The scikit-learn score is negated so **higher scores consistently mean more anomalous**. The estimator is fitted on training rows only, although the scaling step currently uses all rows (see [limitations](#reproducibility-and-limitations)).

### 2. Reconstruction-based anomaly detection

A fully connected **autoencoder** learns to reconstruct each transaction's input features:

```text
166 inputs → 64 hidden units → 8-dimensional bottleneck
           → 64 hidden units → 166 reconstructed inputs
```

It is optimized with **mean squared reconstruction error**, using Adam (`learning rate = 0.001`) for **20 epochs**. A node's reconstruction error serves as its anomaly score. This is a **feature autoencoder**, not a graph autoencoder: it does not explicitly reconstruct edges or aggregate messages from graph neighbors.

A separate **one-class autoencoder** uses the same architecture but is trained on the **23,510 normal training transactions only**, testing whether learning normal feature patterns improves anomaly separation.

### 3. Supervised feature baseline

**Logistic Regression** is trained on labeled training transactions using `class_weight="balanced"` to address class imbalance. Its predicted probability of class `1` is the anomaly score. Because it is feature-based, this model does not directly perform graph message passing.

### 4. Self-supervised graph representation learning

The notebook trains a **64-dimensional embedding for each node ID** using a **link-prediction pretext task**. Positive pairs are observed graph edges; negative pairs are generated by pairing sources with randomly sampled destination IDs. The dot product of two learned embeddings is optimized through binary cross-entropy for **five epochs**. A **LOF detector** is then fitted to the resulting node embeddings to generate anomaly scores.

**Important distinction:** this implementation learns *trainable node-ID embeddings* from observed links. It is **not** a message-passing GNN and does not learn an inductive encoder for unseen node features. Its use of the full edge list is discussed under limitations.

### 5. Graph-aware supervised GCN

The central graph-learning model is a **two-layer GCN implemented directly in PyTorch** using sparse matrix multiplication; it does not rely on PyTorch Geometric. The model:

1. Builds a symmetrically normalized sparse adjacency matrix with self-loops.
2. Aggregates neighboring features and projects them to **64 hidden dimensions**.
3. Applies **ReLU** and **dropout (`0.5`)**.
4. Performs a second graph aggregation and outputs **one logit per node**.
5. Applies a sigmoid at scoring time to obtain an estimated probability of illicit activity.

Training uses **Adam**, a learning rate of **0.01**, weight decay of **0.0005**, and `BCEWithLogitsLoss` with a positive-class weight derived from the training split. Training is capped at **200 epochs**; validation Average Precision is checked at epoch 1 and every 10 epochs thereafter, with the best checkpoint restored. A patience parameter is present, although the saved execution ran through epoch 200 without triggering early stopping.

The notebook creates three adjacency matrices: **train-only**, **train + validation**, and **full labeled graph**. The training loss uses training labels and the train-only adjacency; validation for checkpoint selection uses the train + validation adjacency; final reported GCN scores are computed using the full adjacency. This last detail affects how its reported metrics should be interpreted.

## Evaluation protocol

All methods produce one anomaly score per node, with larger scores interpreted as more anomalous.

**Threshold-independent ranking metrics**

- **ROC-AUC:** discrimination between illicit and licit transactions across thresholds.
- **Average Precision (AP):** summarizes the precision–recall ranking; especially informative with a rare positive class. The notebook names this `PR_AUC_test`, but computes it with `sklearn.metrics.average_precision_score` (AP, not trapezoidal PR-AUC).

**Threshold-dependent decision metrics**

- **Precision:** proportion of flagged transactions that are actually labeled illicit.
- **Recall:** proportion of illicit transactions successfully flagged.
- **F1:** harmonic mean of Precision and Recall.

For each method, candidate thresholds are taken from **unique validation scores**. The threshold maximizing validation F1 is selected and then applied to the test scores. The shared evaluation helper calculates test ROC-AUC and AP directly from continuous scores, and test Precision/Recall/F1 using this selected threshold. A random-score baseline is also evaluated.

> **Caution:** Using validation labels to select thresholds is appropriate, but it does not by itself guarantee freedom from information leakage: feature preprocessing and graph construction must also respect the chronological split.

## Experimental results

The table reproduces the **saved test-set outputs in the uploaded notebook**. These values were **not independently rerun or audited**; see the limitations before using them as benchmark claims.

| Method | Learning setting | Graph links used directly? | ROC-AUC | Average Precision | Precision | Recall | F1 |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: |
| **Two-layer GCN** | Supervised | **Yes** | 0.8217 | **0.4571** | **0.5333** | 0.4654 | **0.4971** |
| LOF (node features) | Unsupervised | No | 0.6687 | 0.1797 | 0.0761 | 0.6840 | 0.1369 |
| Logistic Regression | Supervised | No | **0.8263** | 0.1655 | 0.1597 | 0.7123 | 0.2609 |
| Link-prediction embeddings + LOF | Self-supervised | Yes, for embeddings | 0.4489 | 0.0503 | 0.0568 | 0.9984 | 0.1075 |
| One-Class Autoencoder | One-class | No | 0.2830 | 0.0375 | 0.0569 | 1.0000 | 0.1076 |
| kNN distance | Unsupervised | No | 0.2892 | 0.0374 | 0.0569 | 1.0000 | 0.1076 |
| Standard Autoencoder | Reconstruction-based | No | 0.2704 | 0.0370 | 0.0569 | 1.0000 | 0.1076 |

For reference, a random-score baseline in the notebook has **ROC-AUC ≈ 0.4900** and **AP ≈ 0.0551**. The positive-class rate in the test split is **5.69%**, providing a useful reference level for AP under random ranking.

## Interpretation

The **GCN reports the strongest Average Precision and F1** in this experiment, suggesting that including observed transaction connectivity can be useful for prioritizing illicit transactions. Logistic Regression achieves the highest **ROC-AUC**, illustrating why no single metric captures all aspects of model behavior.

The feature-only autoencoders and kNN distance baseline perform poorly, and their validation-selected thresholds lead to near-universal positive predictions on the test set. This does **not** establish that reconstruction- or distance-based methods are generally unsuitable for fraud detection; it indicates that these particular configurations do not discriminate effectively on this processed dataset. LOF performs substantially better than the other unsupervised feature-only baselines by ranking locally sparse observations.

**These outcomes should be treated as exploratory, not as evidence of deployment readiness or a rigorous leakage-free comparison.** Several components use information from outside the training period, and the final GCN scores are computed under a transductive graph setting.

## Running the notebook

### Environment

- Python 3.10+ is a reasonable starting point; the notebook does **not** pin an exact tested Python or package version.
- Main packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, and Jupyter Notebook.
- The saved output indicates execution on **CPU**. A compatible GPU may accelerate PyTorch training but is not required by the code.
- Internet access is needed to obtain the dataset unless the CSV files are already available locally.

### Kaggle (recommended for the unmodified notebook)

1. Open or upload `Anomaly Detection in Finance (BTC) (1).ipynb` in Kaggle Notebooks.
2. Add the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) as an input.
3. Check that the files are located at the path expected in the code:

   ```python
   DATA_PATH = "/kaggle/input/datasets/organizations/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/"
   ```

4. Run the cells in order, from imports and data loading through the final results table. If Kaggle mounts the dataset under a different directory, update `DATA_PATH` accordingly.

### Local Jupyter environment

Create a virtual environment and install the notebook dependencies:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib scikit-learn torch notebook
jupyter notebook "Anomaly Detection in Finance (BTC) (1).ipynb"
```

Download and extract the Elliptic CSVs locally, for example:

```text
project/
├── README.md
├── Anomaly Detection in Finance (BTC) (1).ipynb
└── data/
    └── elliptic_bitcoin_dataset/
        ├── elliptic_txs_features.csv
        ├── elliptic_txs_edgelist.csv
        └── elliptic_txs_classes.csv
```

Then change the notebook's data path to:

```python
DATA_PATH = "./data/elliptic_bitcoin_dataset/"
```

The code uses a dataset-specific absolute Kaggle path by default; local execution will not work unchanged until that path is updated. File paths are resolved relative to the notebook's current working directory.

## Reproducibility and limitations

The following are **implementation-level caveats observed in the notebook**, not speculative drawbacks of graph ML in general:

1. **Scaling leakage:** `StandardScaler` is fitted to `X` before extracting train/validation/test rows. To evaluate future transactions fairly, fit the scaler on `X[train_idx]` only and transform validation/test with the fitted scaler. The learned embedding scaler is also fitted to all node embeddings.
2. **Transductive kNN:** the kNN reference set contains all labeled transactions, including validation/test rows. A strictly historical baseline should fit its neighbor index only on training data (and adopt an explicit historical updating policy if desired).
3. **Self-supervised future-edge exposure:** link-prediction embeddings are learned from the full retained edge list. The negative sampler also draws arbitrary node pairs without excluding observed positive edges. Restrict training edges chronologically and verify negatives against known edges.
4. **GCN validation contamination at final scoring:** the training checkpoint uses train-only and train+validation adjacency appropriately, but `evaluate_scores` selects its validation F1 threshold from `gcn_scores` calculated using `A_full`. Thus, later/test-period graph connectivity can influence validation scores and threshold selection. A stricter procedure should select the threshold from scores computed using `A_trainval`, then evaluate test nodes using only edges observable at the test cutoff.
5. **Topology changes after label filtering:** unknown-labeled transactions are removed *before* building graph edges, deleting paths and neighbors that existed in the original network. This is a deliberate notebook simplification, not a faithful representation of the full transaction graph.
6. **Directed edges are symmetrized:** the GCN treats transaction links as undirected, discarding transfer direction. Direction-aware or temporal graph methods may better represent the financial process.
7. **Limited reproducibility controls:** `random` and NumPy are seeded (`42`), but PyTorch is not explicitly seeded; environment versions are not pinned, and model initialization and results may vary between executions.
8. **No uncertainty estimates:** the saved results report a single execution, without repeated random seeds, confidence intervals, or formal significance testing. There is no comprehensive hyperparameter search or calibration study.
9. **Label and task interpretation:** the dataset's `illicit` label is used as the positive anomaly class; this is a proxy for illicit-transaction detection, not proof of fraud for an individual real-world transaction. Performance on this dataset does not establish production screening capability.

Because of these issues, **do not describe the reported scores as leakage-free, production-grade, or externally validated**. Re-running the experiment after the relevant corrections may change the relative ranking of methods.

## Suggested extensions

For a stronger research-oriented version of this project:

- **Leakage-controlled benchmark:** move all learned preprocessing inside the train split; create stage-specific train/validation/test graph views; report results again on genuinely future observations.
- **Topology-aware baselines:** compare with degree, local clustering, PageRank, and neighbor-aggregation features using tabular models.
- **Richer GNN models:** add GraphSAGE, GAT, and edge-aware or directed GNN variants under a consistent split and optimization budget.
- **Ablation studies:** remove edges, perturb edge density, compare feature-only and graph-aware variants, and quantify how much connectivity adds over node features.
- **Temporal analysis:** measure drift across time steps and assess performance at different anomaly prevalences.
- **Robustness and uncertainty:** evaluate multiple seeds, confidence intervals, class-prior shifts, threshold sensitivity, and score calibration.
- **Interpretability:** investigate the neighbors, subgraphs, and learned features contributing to high-risk flags.

These are **proposed future improvements**, not features implemented in the supplied notebook.

## Data attribution and author

**Data:** [Elliptic Bitcoin Dataset — Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Please consult the dataset page for its usage terms and citation requirements. The dataset is distributed separately and is not included in this README.

**Notebook author:** **Shayan Rokhva** (2026)

This README documents the methods and saved results of the supplied notebook. No software license is asserted here; add a repository license only if you intend to grant corresponding reuse rights.
