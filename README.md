# MA2C-GNN
This is the Pytorch implementation of Multiple Adaptive Aggregation Channels for GNN-based fraud detection models, named MA2C-GNN for short. The overall view of MA2C-GNN framework comes in the below:

![MA2C-GNN Framework](https://github.com/FraudDetectionModel/MA2C-GNN/assets/136766753/d13de2d9-117a-428d-82a1-983ed51e476b)

# Dependencies
- Python >>> 3.10.9

- Pytorch >>> 2.0.1

- DGL >>> 1.1.1

# Usage
- src: Includes all code scripts, i.e., data preparation, model, and training.
- data: Includes original datasets and dgl graphs for both datasets:
     - YelpChi.zip: Contains hotel and restaurant reviews filtered (spam) and recommended (legitimate) by Yelp.
     - Amazon.zip: Contains product reviews under the Musical Instruments category.
     - yelp.dgl: The dgl graph for original data of YelpChi.
     - amazon.dgl: The dgl graph for original data of Amazon.

# Model Training
