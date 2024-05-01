# Objective

This project aims to construct a multi-task learning model capable of predicting both house prices and categorizing houses by their features using deep learning techniques. Utilizing the PyTorch Lightning framework, the model handles both regression (house price prediction) and classification (house categorization) tasks within a unified architecture. By integrating advanced optimization tools like Optuna, the model dynamically fine-tunes hyperparameters to achieve optimal performance. This approach is particularly valuable for businesses in the real estate sector, enabling them to enhance property valuation accuracy and market segmentation. Accurate price predictions and effective categorization can help real estate professionals make informed investment decisions, optimize their marketing strategies, and improve customer targeting.

# Data Description:

The dataset employed in this project is the "House Prices - Advanced Regression Techniques" from Kaggle, which includes various attributes of houses that influence their market prices and categorization. Key features range from basic attributes such as 'MSSubClass' and 'MSZoning', which describe the type of dwelling and zoning classification, to more complex features like 'Neighborhood' and 'YearBuilt', reflecting the physical location and age of the properties. Detailed descriptions of each attribute can be explored further at [Kaggle Competition: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)


#Data Preprocessing/Feature Engineering:
Key preprocessing steps included setting the 'Id' column as an index for direct house referencing and transforming certain numerical identifiers such as 'MSSubClass' into categorical variables, acknowledging their non-numeric nature. To handle missing data, strategies like removing columns with excessive missing values (e.g., 'Alley', 'PoolQC') and imputing others using the mode for categorical and median for numerical entries were implemented. Additionally, the detection and curation of outliers through the interquartile range method primarily focused on the 'SalePrice' to prevent skewed analyses. A pivotal feature engineering step was the creation of a new variable, 'House Category', derived from the house's style, type, and age, which facilitated the model's classification task. The final preprocessing pipeline employed one-hot encoding for categorical variables and standardization for numerical variables to ensure consistent data scale across all inputs.

# Modelling:

The focus was on developing a robust multi-task learning framework capable of simultaneously predicting house prices and categorizing houses. This required the creation of a shared architecture that could efficiently handle both regression and classification tasks, integrated through PyTorch Lightning to leverage its comprehensive suite of deep learning tools.

**Model 1: ReLU and Adam Integration**
Model 1 was constructed using shared layers that included sequences of linear layers, ReLU activations, and dropout layers. These shared components were crucial as they allowed for feature extraction that benefited both the price prediction and house categorization tasks. Specifically, the price prediction sub-model extended these shared layers with additional linear transformations, culminating in an output layer that directly predicted house prices. Conversely, the classification sub-model branched off similarly but ended with a sigmoid activation to output probabilities for house categories. The choice of ReLU helped in mitigating the vanishing gradient problem, making the model efficient during training across multiple epochs. Adam optimizer was utilized for its adaptive learning rate capabilities, making it ideal for handling different data distributions and scales efficiently.

**Model 2: Tanh and SGD Customization**
Model 2 differentiated itself by incorporating Tanh activations within its architecture, which normalized outputs between -1 and 1, aiding in handling features where normalization was beneficial. This model also used the Stochastic Gradient Descent (SGD) optimizer, which, while simpler than Adam, offered robustness through its straightforward update rules. This model configuration was particularly focused on capturing nonlinear relationships effectively. The use of Xavier initialization complemented the Tanh activations by maintaining the variance of activations across layers, which is critical in preventing the vanishing or exploding gradients problem.

**Model 3: PReLU and RMSProp Optimization**
Model 3 adopted a more complex and adaptive approach with PReLU activations and the RMSProp optimizer. PReLU, a variant of ReLU, introduces learnable parameters that allow each neuron to adapt its activation function, potentially capturing more complex patterns in the data. RMSProp, different from both Adam and SGD, adjusts learning rates based on a moving average of squared gradients, offering advantages in scenarios with noisy or sparse gradient issues. This model's architecture aimed at robustness and adaptability, particularly in datasets with significant variability or anomalies.

**Hyperparameter Tuning with Optuna**
Across all models, Optuna was employed to optimize hyperparameters dynamically. This involved adjusting parameters such as the number of neurons in hidden layers, dropout rates to prevent overfitting, and learning rates to ensure effective convergence. The tuning process was integral to refining each model's configuration based on the validation loss, facilitating the identification of the most effective model setups through systematic trials.

**Model Evaluation and Testing Strategy**
The evaluation of the models was meticulously carried out using a range of metrics appropriate for both regression and classification tasks, including MSE, Accuracy, F1 Score, Precision, Recall, and R2 Score. These metrics provided insights into each model's performance across the training and validation phases. PyTorch Lightning's Trainer API was utilized for training with features like model checkpoints, early stopping to prevent overfitting, and detailed logging for real-time performance tracking.

Testing was conducted on an unseen test dataset, ensuring models were not just evaluated on their performance during training or validation but also on how well they could generalize to new data. This testing phase was critical in assessing the real-world applicability of the multi-task learning models developed in the project

# Results

## Best Model Results
The table below summarizes the performance of the best model (Model 1) across various metrics during the training and validation phases. This model excelled due to its robust architecture and effective use of optimization and activation functions, which contributed to high accuracy and generalization capabilities:

| Metric         | Training | Validation |
|----------------|----------|------------|
| Accuracy       | 94.98%   | 94.76%     |
| Precision      | 94.00%   | 91.83%     |
| Recall         | 93.67%   | 91.82%     |
| F1-Score       | 94.98%   | 94.73%     |
| Total Loss     | 0.32     | 0.468      |
| MSE Regression | 0.0155   | 0.1722     |
| R2 Score       | 0.7715   | 0.8219     |

### Insights from the Results

- **Accuracy and F1-Score:** Both high accuracy and F1-score across training and validation phases indicate that the model is well-tuned and capable of delivering consistent predictions, effectively balancing precision and recall.
- **Precision and Recall:** The precision and recall values close to or above 90% show that the model accurately identifies relevant instances and categorizes the house types with minimal error, making it reliable for practical use.
- **Total Loss:** The low total loss during training, which slightly increases during validation, suggests that while the model is fitted well to the training data, it also maintains a reasonable generalization to unseen data without overfitting.
- **MSE and R2 Score:** The Mean Squared Error (MSE) and R2 scores highlight the model's capability to predict house prices with high accuracy, where a higher R2 score in validation than training indicates an improvement in generalizing the regression task beyond the training dataset.

This detailed performance analysis confirms that Model 1, through its architectural choices and hyperparameter settings, effectively addresses both the regression and classification tasks, making it a robust solution for predicting house prices and categorizing houses based on their characteristics.


