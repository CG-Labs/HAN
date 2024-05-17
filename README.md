# Paper
https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph
# HAN

The source code of Heterogeneous Graph Attention Network (WWW-2019).

The source code is based on [GAT](https://github.com/PetarV-/GAT)

# Reference

If you make advantage of the HAN model or use the datasets released in our paper, please cite the following in your manuscript:

```
@article{han2019,
title={Heterogeneous Graph Attention Network},
author={Xiao, Wang and Houye, Ji and Chuan, Shi and  Bai, Wang and Peng, Cui and P. , Yu and Yanfang, Ye},
journal={WWW},
year={2019}
}
||||||| parent of 73d3c53c (Update model, state machine, and import script; add structured CV data)
## Visualization
To visualize the model's embeddings:
1. Run the `ex_acm3025.py` script to train the model and generate embeddings.
2. Use the `tsne_visualization.py` script to create a t-SNE plot of the embeddings.

## Installation and Usage
To install and set up the HAN project with the new features, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/CG-Labs/HAN.git
   ```
2. Navigate to the cloned repository directory:
   ```
   cd HAN
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Follow the Neo4j Setup instructions above to configure the graph database.
5. Place any CV documents you wish to process in the `data` directory.
6. Run the `process_cv_data.py` script to preprocess the CV data.
7. Run the `ex_acm3025.py` script to train the model.
8. Use the `tsne_visualization.py` script to visualize the embeddings with t-SNE.

## Workflow
The workflow has been updated to include the new steps for CV processing and Neo4j integration. Refer to the updated mermaid graph below for a visual representation of the workflow.

```mermaid
graph TD;
    A[CV Document] -->|Process| B[Feature Vectors & Adjacency Matrix];
    B -->|Integrate| C[Neo4j Database];
    C --> D[Model Training];
    D --> E[Model Evaluation];
    E --> F[t-SNE Visualization];
=======
## Visualization
To visualize the model's embeddings:
1. Run the `ex_acm3025.py` script to train the model and generate embeddings.
2. Use the `tsne_visualization.py` script to create a t-SNE plot of the embeddings.

## Installation and Usage
To install and set up the HAN project with the new features, including Neo4j integration and state machine functionality, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/CG-Labs/HAN.git
   ```
2. Navigate to the cloned repository directory:
   ```
   cd HAN
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Follow the Neo4j Setup instructions above to configure the graph database.
5. Place any CV documents you wish to process in the `data` directory.
6. Run the `process_cv_data.py` script to preprocess the CV data. This script takes a CV document as input and outputs feature vectors and an adjacency matrix.
   ```
   python process_cv_data.py --cv <path_to_cv_document>
   ```
7. Run the `gnn_model.py` script to initialize the graph neural network model. This script requires the feature vectors and adjacency matrix as input.
   ```
   python gnn_model.py --features <path_to_feature_vectors> --adjacency <path_to_adjacency_matrix>
   ```
8. Execute the `state_machine.py` script to process user queries and perform actions based on the system's state. This script accepts user queries as input and outputs the system's response.
   ```
   python state_machine.py --query "<user_query>"
   ```

For detailed usage of each script, including the expected input and output, refer to the comments within the script files.

## Workflow
The workflow has been updated to include the new steps for CV processing and Neo4j integration. Refer to the updated mermaid graph below for a visual representation of the workflow.

```mermaid
graph TD;
    A[CV Document] -->|Process| B[Feature Vectors & Adjacency Matrix];
    B -->|Integrate| C[Neo4j Database];
    C --> D[Model Training];
    D --> E[Model Evaluation];
    E --> F[t-SNE Visualization];
```

## Requirements
The `requirements.txt` file lists all the necessary packages to run the HAN project with the new features. Install all the dependencies listed in this file to ensure proper functionality of the project.

## Testing Procedures and Results
To ensure the integrity and performance of the HAN model, comprehensive testing procedures have been implemented:
1. Unit tests for individual components (data preprocessing, model training, Neo4j integration) are located in the `tests` directory. To run these tests, use the following command:
   ```
   python -m unittest discover -s tests
   ```
2. Integration tests to verify the end-to-end workflow from CV processing to graph database integration. These tests can be found in the `tests/integration` directory.
3. Performance tests to evaluate the model's accuracy and efficiency with different sizes of datasets. The results are documented in the `tests/performance` directory.

The test results indicate that the model performs well with the provided datasets and is robust to variations in CV document formats. The t-SNE visualization provides a clear representation of the model's ability to differentiate between various features in the data. Detailed test reports can be found in the `tests/reports` directory.

## State Machine and Predictive Analysis
The HAN project now includes a state machine designed to enhance system autonomy by interpreting user input, understanding intent, and orchestrating tasks. The state machine operates in several states, including data retrieval, data analysis, and prediction, to autonomously execute actions based on user queries and data insights.

### State Machine Functionality
- **Data Retrieval**: The system retrieves relevant data from the Neo4j database, preparing it for analysis.
- **Data Analysis**: Utilizing the graph neural network model, the system analyzes the data to identify patterns and insights.
- **Prediction**: Based on the analysis, the system makes predictions, such as forecasting trends or identifying opportunities and threats within datasets.

The state machine's logic is implemented in the `state_machine.py` file, which contains methods for each state and action. This allows the system to process complex queries, such as predicting Bitcoin prices at the end of 2024, by leveraging the trained graph neural network model and the structured data within Neo4j.

### Predictive Analysis Process
The predictive analysis component uses the graph neural network model to analyze data and make forecasts. The model is trained on datasets provided by the user, learning to recognize patterns and correlations that can be used to make predictions about future data points. The system is designed to handle various data formats, including CSV, DOC, and PDF, allowing for a wide range of input data types. Once trained, the model can be used to answer complex queries and make predictions, such as forecasting Bitcoin prices at the end of 2024, by leveraging the structured data within Neo4j and the insights gained from the analysis.

## Future Work and Known Limitations
Future work on this project may include:
- Expanding the model's language capabilities to handle more nuanced and complex language constructs.
- Enhancing the visualization component to offer more interactive and detailed views of the graph data.
- Improving the scalability of the Neo4j integration to handle larger datasets more efficiently.

1. ACM_3025 in our experiments is based on the preprocessed version ACM in other paper (\data\ACM\ACM.mat). Subject is just like Neural Network, Multi-Object Optimization and Face Recognition. In ACM3025, PLP is actually PSP. You can find it in our code.
2. In ACM, train+val+test < node_num. That is because our model is a semi-supervised model which only need a few labels to optimize our model. The num of node can be found in meta-path based adj mat.
3.  "the model can generate node
embeddings for previous unseen nodes or even unseen graph" means the propose HAN can do inductive experiments. However, we cannot find such heterogeneous graph dataset. See experiments setting in Graphsage and GAT for details, especially on PPI dataset.
4. meta-path can be symmetric or asymmetric. HAN can deal with different types of nodes via project them into the same space.
5. Can we change the split of dataset and re-conduct some experiments? of course, you can split the dataset by yourself, as long as you use the same split for all models.
6. How to run baseline (e.g., GCN) and report the best performance of baselines? Taking ACM as an example, we translate heterogenesous graph into two homogeneous graphs via meta-path PAP&PSP. For PAP based homogeneous graph, it only has one type of node paper and two paper connected via PAP. Then, we run GCN on two graphs and report the best performance. Ref https://arxiv.org/pdf/1902.01475v1.pdf and http://web.cs.wpi.edu/~xkong/publications/papers/www18.pdf
7. Several principles for preprocess data. 1）Extract nodes which have all meta-path based neighbors. 2）Extract features which may meaningful in identifying the characteristics of nodes. For example, if all nodes have one feature, this feature is not meaningful. If only several nodes have one feature, this feature is not meaningful. 3) Extract balanced node label which means different classes should have almost the same number of node. For k classes, each class should select 500 nodes and label them, so we get 500\*k labeled nodes.

# Datasets

Preprocessed ACM can be found in:
https://pan.baidu.com/s/1V2iOikRqHPtVvaANdkzROw
提取码：50k2

https://bupteducn-my.sharepoint.com/:u:/g/personal/jhy1993_bupt_edu_cn/EfLZcHE2e4xBplCVnzcJbQYBurNVOCk7ZIne2YsO3jKbSw?e=vMQ18v

Preprocessed DBLP can be found in:
https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg
提取码：6b3h

https://bupteducn-my.sharepoint.com/:u:/g/personal/jhy1993_bupt_edu_cn/Ef6A6m2njZ5CqkTN8QcwU8QBuENpB7eDVJRnsV9cWXWmsA?e=wlErKk

Preprocessed IMDB can be found in:
链接:https://pan.baidu.com/s/199LoAr5WmL3wgx66j-qwaw  密码:qkec


# Run
Download preprocessed data and modify data path in def load_data_dblp(path='/home/jhy/allGAT/acm_hetesim/ACM3025.mat'):

python ex_acm3025.py

# HAN in DGL
https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
