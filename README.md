(pdf is more legible)

**Credit Card Fraud Risk Assessment Using Machine Learning**
DATA 543
Mihir Harshe, Joshua Senthil, Yanqing Su

**Abstract**
Credit card fraud has evolved into a rising global challenge as more transactions shift to a virtual landscape and become easier to scale. This project looks at how well different machine learning models can detect fraudulent transactions and estimate their associated risk. We worked with two large datasets, including one with about 500,000 transactions and another European dataset with anonymized features. One of our biggest concerns going in was that fraud cases are extremely rare compared to traditional transactions, thus many models tend to ignore fraud entirely. To address this, we tested several approaches, including class weighting and SMOTE, which generates synthetic fraud examples. Our results show that most standard models perform poorly unless the imbalance is properly accounted for. The best-performing approach was Logistic Regression combined with SMOTE, which achieved the most balanced performance across all metrics. We also briefly explored GANs for generating artificial data, but found that it was not effective in this setting due to limited fraud examples. Overall, this project shows that handling imbalance is critical, and that in fraud detection, identifying fraud (recall) matters more than avoiding false alarms (precision).

**Introduction**
**Background**
Credit cards are a core part of how people make transactions today, mostly because of how effortless and convenient they are. At the same time, this convenience has made fraud more common and easier to carry out at scale. Fraud not only affects individuals, but also creates major costs for financial institutions.
**Problem Statement**
The main challenge in fraud detection is that fraudulent transactions make up a very small portion of the data (typically less than 2%). This creates a strong imbalance that causes most models to predict non-fraud by default.
Fraud itself represents the hazard
Financial systems and customers represent the exposure
Model errors (false positives and false negatives) represent vulnerability
**Objectives**
In this project, our goal is to:
Compare machine learning models for fraud detection
Evaluate methods for handling imbalanced data
Quantify fraud risk using hazard, exposure, and vulnerability
Analyze the tradeoff between type I and type II errors
Identify patterns linked to fraud

**Data Description**
**Data Sources**
The first data source is used for our risk and ML analysis. It is called Credit Card Fraud 2025. It is five thousand credit card transactions in 2025 gathered by Data Analyst Prince Rajak for Machine Learning application. He has created over 90 different datasets to be used for competition and machine learning applications and 8.82 usability, so we decided it was one of the stronger datasets found on Kaggle. The data is certain around time, geography and transaction logistics.
 https://www.kaggle.com/datasets/prince7489/credit-card-fraud-2025 
The second data source was called Credit Card Fraud Detection by the Wordline and Université Libre de Bruxelles ML group. The data is two hundred eighty thousand transactions from Europe in the earlier part of this decade. The data is mostly anonymized, so we wanted to use this by deciding a model to use without the bias of context. However, it was ultimately used for our exploration of the GANs model.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 
**Features**
Credit Card Fraud 2025: In terms of contextual data, features include Transaction Date, Country and International Transactions. In terms of logistics, we have the amount, merchant, merchant category, type of card, transaction type (Online, Atm, etc.). We also have the fraud indicator, transaction id, and customer id.
Credit Card Fraud Detection: Most of the data is reduced and anonymized through Principal Component Analysis. The features then are the time of the transaction, the amount, the fraud indication, and twenty-eight anonymized variables. 
**Data Challenges**
Both datasets are highly imbalanced as the amount of fraudulent data is relatively small in comparison to the non-fraudulent data. As a result, that will add more layers and implications when going through the project.
With the Credit Card Fraud Detection, having PCA data limits the contexts so that may hinder some of interpretability of the performance in terms of what exactly is influencing that decisions or precision.

**Methodology**
**Data Preparation**
We loaded the dataset and used fraud_flag as the target variable. Identifier columns (e.g., transaction and customer IDs) were removed since they do not help prediction. We used pandas for data preparation and scikit-learn and numpy for the visualization aspect, and our numerical calculations respectively.
The data was split into features (X) and labels (y), and categorical variables were converted using one-hot encoding. As expected, the dataset was highly imbalanced, with very few fraud cases.
**Train-Test Split and Scaling**
We split the data into 80% training and 20% testing, using stratification to preserve class imbalance.
For Logistic Regression models, we applied standardization using StandardScaler to make sure that features were on the same scale.
**Models Tested**
We tested five models to compare different approaches to imbalance:
Logistic Regression (Baseline): No adjustments; used as a reference point
Logistic Regression (Balanced): Added class weights to highlight fraud cases
Random Forest: Captures more complex patterns but trained on imbalanced data (this is why we ended up choosing a different model)
SMOTE + Random Forest: Balanced the training data using synthetic fraud samples
Logistic Regression + SMOTE: Combined balancing with a simple model (best performer)
**Evaluation**
Models were evaluated using:
Precision (false positives)
Recall (fraud detected — most important)
F1-score (balance of both)
We also used confusion matrices, ROC curves, and precision–recall curves to better understand performance.
**Handling Imbalance**
We compared two main strategies:
Class weighting (penalizing fraud misclassification)
SMOTE (creating synthetic fraud examples)
This allowed us to test whether models learn better from reweighted data vs balanced data.
**Risk Perspective**
Performance was interpreted through risk:
Low recall → missed fraud → financial loss
Low precision → false alarms → customer frustration
Because missing fraud is more costly, recall was prioritized.

**Data Preprocessing**
We started by cleaning the data and making sure all features were in a usable format. After that, we split the data into training and testing sets. We also applied basic scaling where needed and made sure there was no overlap between training and test data.
**Handling Class Imbalance**
We tried several different approaches:
Leaving the data as-is (baseline)
Using class weights to penalize mistakes on fraud cases
Bootstrapping to resample the data
Using SMOTE to generate synthetic fraud examples
**Models Implemented**
We tested five models:
Logistic Regression (baseline)
Logistic Regression with class weights
Random Forest
Gradient Boosting
Logistic Regression with SMOTE
**Evaluation Metrics**
We evaluated models using standard metrics such as precision, recall, F1-score, and ROC-AUC. However, in this project, these metrics are directly tied to **risk interpretation:**
- **Recall (Type II Error)** → measures how much fraud is missed
- **Precision (Type I Error)** → measures how often legitimate transactions are flagged
- **F1-score** → balances both types of errors
From a risk perspective:
Type I errors (false positives) impact **customer experience and trust**
Type II errors (false negatives) lead to **direct financial loss**
Because financial loss is typically more severe, recall is prioritized.

**Results**
**Model Performance Comparison**
The baseline Logistic Regression model had high accuracy, but it completely failed to detect fraud, with 0% recall. This shows how misleading accuracy can be in imbalanced problems.
Adding class weights improved recall to around 51%, but precision dropped significantly, meaning the model flagged many normal transactions as fraud.
Random Forest and Gradient Boosting models didn’t perform well either. While they are more complex, they still struggled to identify fraud because they didn’t see enough examples of it during training.
The best results came from Logistic Regression with SMOTE, which improved recall to about 52% and had the strongest overall balance between metrics.
**Key Findings**
Models fail if imbalance is not addressed
Improving recall reduces financial risk but increases customer friction
SMOTE helps models better capture fraud patterns
Simpler models outperform complex ones when aligned with the risk structure
**Threshold Tuning**
Lowering the classification threshold increases recall, meaning fewer fraudulent transactions are missed. However, this increases false positives.
From a risk perspective:
Higher recall → reduces **financial exposure**
Lower precision → increases **customer inconvenience and potential churn**
This tradeoff reflects the balance between **monetary loss and reputation risk.**

**Discussion**
**Why Some Models Failed**
Most models struggled because they were trained on data dominated by normal transactions. As a result, they learned to predict “non-fraud” almost all the time.
**Strengths & Weaknesses of Each Model**
Logistic Regression: simple but ineffective without adjustments
Balanced Logistic Regression: better recall but too many false positives
Random Forest: captures more complexity but still misses fraud
Gradient Boosting: underperformed in this case
SMOTE + Logistic Regression: best balance overall
**Key Variables Driving Fraud**
Important features included transaction amount, merchant ID, distance from home, and time of day. This suggests that fraud is more about unusual behavior patterns than any single feature.

**Advanced Exploration: GANs**
**Motivation**
One of the hazards mentioned for Credit Card Fraud is Data leakage. When we use this data for ML applications, it creates a new infrastructure that hackers can target. As a result, we are reciprocating the hazard potentially. As a result, many corporations are hesitant to scale processes towards implementing ML due to this reason. The downside of this tradeoff is that performance is reduced due to underfitting. Secondly, the imbalance impedes progress in this field. We wanted to explore a potential solution of creating synthetic data using a machine learning model. We decided on the GANs model due to the familiarity of it.
**GAN Framework**
The GANs model is a Generative Adversarial Networks where there are two Neural Networks, usually used for image creation, but we are using it for tabular data. The discriminator gets the real data and makes a noise vector for the generator. THe generator then makes synthetic data and the discriminator randomly takes either the real data or synthetic data to guess if it's real. The discriminator will feed this as a noise vector to the generator again and again until the generator makes data that is indistinguishable.
**Limitations Observed**
In our case, GANs did not perform well:
Fraud is only 0.2% of the data, so the imbalance learns from the safe data and mimics this more likely. The generator is too safe
Underfitting fraud data leads to inefficiencies found in the model.
The reduction of context from PCA data hurts performance of the model.
Based on the performance at each epoch, the discriminator became too powerful for the generator.
**Conclusion on GANs**
The application of Gans has to be used in conjunction with another model, not by itself. The best use is using the feature classification with another classifier to make it more robust.

**Risk & Ethical Considerations**
Handling sensitive financial data comes with privacy risks as leaking the data will result in fraud, theft, or exposure sensitivity, thus there’s a trade off  
A type one error would be calling an actual transaction as fraudulent (False Positive). The outcome would result in a person's transaction getting blocked, resulting in inconvenience and frustration.
On the other hand, we can have a type two error, which is allowing a fraudulent transaction to go. The outcome would result in the bank losing money due to chargeback since this is the standard policy. This also may cause clients to see the money leave all of a sudden.

**Recommendations**
Use Logistic Regression with SMOTE as a baseline
Focus on maximizing recall
Gather more fraudulent data to counter the imbalance issue
Create more robust models by ensemble techniques or stacking
Increase prevention through multi-authentication on credit cards

**Limitations**
Hindrance of context because of privacy inherently disrupts performance.
Some hardware limits decrease the potential of models such as asynchronous computing or time used on an epoch.
Imbalance hurts the performance of fraudulent data.
Credit card data is sensitive so gathering much more for an efficient scale is very difficult compared to gathering other data due to clearance issues.

**Future Work**
Add more meaningful features
Gather more data
Try a new combination of activation and functions within models.
Explore hybrid systems
Improve synthetic data generation methods

**Conclusion**
Fraud detection is a difficult problem mainly because of how rare fraud cases are. Our results show that handling imbalance is more important than choosing a complex model. Logistic Regression with SMOTE provided the best results in our case, but no single model is enough on its own. In practice, fraud detection systems need to combine multiple approaches to be effective such as ensemble techniques. Furthermore, in the use of synthetic data, the GANs model falls behind due to similar reasons. In terms of vulnerability, there is no one variable that is a clear indicator of fraud, but the combination of Home distance, merchant mishandling, and hour of time. There's no specific demographic impact. From a risk perspective, false negatives (missed fraud) are far more costly than false positives, as they directly lead to financial loss through chargebacks that mirror typical transaction amounts and volumes. This is real exposure within the risk.

**References**
Cobb, Debbie. “2025 by the Numbers: More Attacks, Slightly More Cards.” FICO, 26 March 2026, https://www.fico.com/blogs/state-card-skimming-us-2025-year-review. Accessed 24 April 2026.
Geeks For Geeks. “Basics of Generative Adversarial Networks (GANs).” GeeksforGeeks, 1 August 2025, https://www.geeksforgeeks.org/machine-learning/basics-of-generative-adversarial-networks-gans/. Accessed 3 May 2026.
Rajak, Prince. “Credit Card Fraud 2025.” Kaggle.com, 3 November 2025, https://www.kaggle.com/datasets/prince7489/credit-card-fraud-2025. Accessed 5 April 2026.
Worldline, and Machine Learning Group of Université Libre de Bruxelles. “Credit Card Fraud Detection.” Kaggle, 3 March 2021, https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Accessed 3 May 2026

