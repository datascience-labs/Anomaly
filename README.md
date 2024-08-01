# Enhancing Multivariate Time Series Anomaly Detection through Causal Disentanglement with Graph Neural Networks

This repo includes a complete framework for multivariate anomaly detection, using a model that is heavily inspired by CDRL4AD(Causally Disentangled Representation Learning for Anomaly Detection).  
For contact, temp@temp.com

## ❗Key Features
1. Causal Discovery:
   - Utilizes an attention mechanism to identify and learn time-lagged causal relationships within multivariate time series data.
   - Constructs a Directed Acyclic Graph (DAG) to represent these causal relationships, enhancing the model's understanding of how past data influences future outcomes.
2. Causally Disentangled Representation (CDR):
   - Employs a multi-head decoder variational autoencoder (VAE) to create causally disentangled representations.
   - Ensures each latent variable aligns with predefined causal relationships, allowing for sophisticated data representation and better interpretability.
3. Node and Edge Correlation Representation (NECR):
   - Encodes variable correlations at both node and edge levels using a dual-level Graph Attention Network (GAT).
   - Learns graph structures by connecting nodes with similar patterns, improving the model’s ability to detect anomalies through structural insights.
4. Temporal Dependency Representation (TDR):
   - Captures sequential relationships within the data, accounting for dependencies over time.
   - Aggregates features from past observations to model temporal dependencies, enhancing the detection of anomalies with temporal context.
5. Root Cause Analysis:
   - Provides clear insights into the root causes of detected anomalies by analyzing variable contributions.
   - Identifies top-K root cause variables, facilitating efficient diagnosis and problem-solving in real-time scenarios.
6. Model Interpretability:
   - Offers t-SNE visualization of causally disentangled representations, illustrating distinct clusters of variable representations.
   - Enhances model transparency by explicitly encoding causal influences and their significance in the representation.
7. Robust Performance:
   - Demonstrates high precision, recall, and F1 scores across diverse datasets, showcasing the model's robustness to different data characteristics and distributions.
   - Includes comprehensive ablation studies to validate the importance of each model component.

## 🖥️ Getting Started

To clone the repo:
~~~bash
git clone https://github.com/datascience-labs/cdrl4ad.git && cd cdrl4ad
~~~

Install dependencies (virtual env is recommended):
~~~bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
~~~

To train:
~~~
python train.py --dataset <dataset> --epochs <epoch>
~~~

## Data
We used the SWaT, HAI, SMD, and PSM datasets. The originals of each dataset can be requested through the links below.  

| Dataset | Source |
|---------|--------|
| SWaT    | [iTrust](https://itrust.sutd.edu.sg/) |
| HAI     | [HAI](https://www.kaggle.com/datasets/icsdataset/hai-security-dataset) |
| SMD     | [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) |
| PSM     | [RANSynCoders](https://github.com/eBay/RANSynCoders/tree/main/data) |

## Cite
TBA
