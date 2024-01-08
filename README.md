# Credit Default Predictor - Kaggle Competition

This repository encompasses a comprehensive data analysis project focused on predicting credit default for the Kaggle competition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview). Here's a breakdown of the key steps:

## Handling Missing Values

The project starts by addressing missing values in the dataset sourced from Kaggle. The scikit-learn library is used for median imputation to handle missing values in features.

## Identifying Outliers

Outliers are identified using a common rule based on interquartile range (IQR). The percentage of outliers for specific features is detailed, specific to the credit default prediction task.

## One-Hot Encoding

Categorical features are transformed into binary vectors using the `get_dummies` method of the pandas library. The datasets are aligned to match dimensions, essential for subsequent machine learning tasks.

## Scaling

Feature scaling is performed to bring features within a uniform range (0 to 1) using scikit-learn's `MinMaxScaler`. This ensures that different features do not carry disproportionate weights in the machine learning models.

## Data Visualization

Seaborn, Plotly, and Matplotlib are utilized for visualizing the Kaggle dataset. Anomalies, such as unrealistic values, are visually identified and handled appropriately to enhance the credit default prediction model.

## Description of Algorithms

### Models Used:

The heart of the project lies in employing four classification algorithms - 

1. **Logistic Regression:** A widely applied supervised learning binary classification algorithm, used as the baseline model. It assumes a linear relationship between the feature set.

2. **Random Forest:** A commonly used supervised machine algorithm that combines predictions from multiple individual decision trees to generate a more accurate prediction. It is robust to different data types and can handle large datasets.

3. **LightGBM (Light Gradient Boosting Machine):** Used for efficient and accurate machine learning tasks, known for its speed, memory efficiency, and performance. It is robust to imbalanced data and has low memory usage.

4. **XGBoost (eXtreme Gradient Boosting):** A fast, scalable, gradient-based decision tree machine learning algorithm. It is versatile, working well with extensive/small data and handling both categorical and continuous variables.

## Hyperparameter Tuning

To optimize the performance of the credit default prediction models, hyperparameter tuning is performed. This ensures the models are fine-tuned for accurate predictions.

## Experiment

The experiment delves into a detailed exploration of a diverse dataset provided by Kaggle, including application data, bureau data, previous application data, credit card balance data, POS cash balance data, and installments data.

## Result

The results indicate that feature engineering, combined with feature selection, can achieve comparable model performance with reduced computational costs. Key evaluation metrics specific to the credit default prediction task are presented in the result section.

## Conclusion

Through the project, we gained valuable experience working with a real-world dataset, handling missing values, and employing k-fold cross-validation. The significance of feature engineering and feature selection in improving model performance and computational efficiency became evident. Moving forward, we believe that sampling techniques like SMOTE and ADASYN can further enhance results, addressing the imbalance in the target class. Future work may also explore GPU-accelerated libraries to expedite the experimentation process. Modeling financial data proved to be a valuable learning experience, and the skills acquired will undoubtedly contribute to future projects.

## Dependencies

- Python 3.x
- Scikit-learn
- Pandas
- Seaborn
- Plotly
- Matplotlib
- LightGBM
- XGBoost

## Usage

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Execute the project notebook in a Google Colab environment.

Feel free to explore the code, modify parameters, and experiment further with the Kaggle dataset. Your feedback and contributions are welcome!
