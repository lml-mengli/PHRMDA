# PHRMDA
PHRMDA:Pairwise and Higher-Order Relational Representation Learning for Microbe–Drug Association Prediction


## Overview
PHRMDA (Pairwise and Higher-Order Relational Representation Learning for Microbe–Drug Association Prediction) is designed to address the unequal structural information available to nodes in microbe–drug association (MDA) networks.

Known associations are unevenly distributed across nodes. Highly connected nodes contribute more information during message propagation, whereas nodes with few known associations have limited structural information for representation learning. 
In addition, pairwise associations alone cannot directly capture potential relationships involving multiple nodes.

To address these issues, PHRMDA integrates pairwise and higher-order relationship learning. 
FSAGD constructs auxiliary structural connections and adaptively integrates multi-hop diffusion information, while a learnable hypergraph captures potential higher-order relationship patterns. 
Cross-view contrastive learning further promotes the integration of representations from the two structural views for MDA prediction.

## Result model
The prediction results of the model on the MDAD dataset are available in the `result_MDAD` directory.


## Data
experiments were conducted on three publicly available datasets, including MDAD[1], aBiofilm[2], and DrugVirus[3]. 

[1]	Sun Y-Z, Zhang D-H, Cai S-B et al. MDAD: a special resource for microbe-drug associations, Frontiers in cellular and infection microbiology 2018;8:424.

[2]	Rajput A, Thakur A, Sharma S et al. aBiofilm: a resource of anti-biofilm agents and their potential implications in targeting antibiotic drug resistance, Nucleic acids research 2018;46:D894-D900.

[3]	Andersen PI, Ianevski A, Lysvand H et al. Discovery and development of safe-in-man broad-spectrum antiviral agents, International Journal of Infectious Diseases 2020;93:268-276.


## Training and Evaluation
To run the model with 10-fold cross-validation, execute: python main.py
