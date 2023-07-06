# MA2C-GNN
This is the Pytorch implementation of Multiple Adaptive Aggregation Channels for GNN-based fraud detection models, named MA2C-GNN for short. The overall view of MA2C-GNN framework comes in the below:

![MA2C-GNN Framework](https://github.com/FraudDetectionModel/MA2C-GNN/assets/136766753/d13de2d9-117a-428d-82a1-983ed51e476b)

# Dependencies
- Python >= 3.10.9

- Pytorch >= 2.0.1

- DGL >= 1.1.1

# Usage
- src: Includes all code scripts, i.e., data preparation, model, and training.
- original-data: Includes original datasets, i.e., YelpChi and Amazon.
     - YelpChi.zip: Contains hotel and restaurant reviews filtered (spam) and recommended (legitimate) by Yelp.
     - Amazon.zip: Contains product reviews under the Musical Instruments category.
- dgl-data: Include dgl graph of original datasets, i.e., YelpChi and Amazon. 
     - yelp.dgl: The dgl graph for original data of YelpChi.
     - amazon.dgl: The dgl graph for original data of Amazon.
- dgl-data-completed: Final dgl graphs of YelpChi and Amazon which include the outputs of graph-agnostic edge labeling module, i.e., edge labels and 
                      domination signals for each individual node in the graph.
     - yelp_completed.dgl: The dgl graph of YelpChi dataset which includes generated edge labels and domination signal for a given neighborhood and use for training phase.
     - amazon_completed.dgl: The dgl graph of Amazon dataset which includes generated edge labels and domination signal for a given neighborhood and use for training phase.

# Model Training
To generate the complete version of dgl graphs (i.e. yelp_completed.dgl and amazon_completed.dgl) we need to run the prepare_data.py and train the graph-agnostic edge labeling module by calling the edgeLabelling_train function from train.py. We put the completed version of trained dgl graphs in dgl-data-completed directory. Therefore, you need to Unzip the yelp_completed.dgl.zip and amazon_completed.dgl.zip from dgl-data-completed directory and move them to the src directory for the training phase.

- We take YelpChi as an example to illustrate the usage of repository:
    - Run train.py
    - Load the yelp_completed.dgl 
    - Start training and testing phases 
