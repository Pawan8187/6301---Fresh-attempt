# Credit Line Increase Model Card

### Basic Information

* **Person or organization developing model**: John Doe, `jdoe@gwu.edu`
* **Model date**: October 6, 2022
* **Model version**: 1.0
* **License**: MIT
* **Model implementation code**: [DNSC_6301_Project.ipynb](DNSC_6301_Project.ipynb)

### Intended Use
* **Primary intended uses**: This model is an *example* probability of default classifier, with an *example* use case for determining eligibility for a credit line increase.
* **Primary intended users**: Students in GWU DNSC 6301 bootcamp.
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

### Training Data

* Data dictionary: 

| Name | Modeling Role | Measurement Level| Description|
| ---- | ------------- | ---------------- | ---------- |
|**ID**| ID | int | unique row indentifier |
| **LIMIT_BAL** | input | float | amount of previously awarded credit |
| **SEX** | demographic information | int | 1 = male; 2 = female
| **RACE** | demographic information | int | 1 = hispanic; 2 = black; 3 = white; 4 = asian |
| **EDUCATION** | demographic information | int | 1 = graduate school; 2 = university; 3 = high school; 4 = others |
| **MARRIAGE** | demographic information | int | 1 = married; 2 = single; 3 = others |
| **AGE** | demographic information | int | age in years |
| **PAY_0, PAY_2 - PAY_6** | inputs | int | history of past payment; PAY_0 = the repayment status in September, 2005; PAY_2 = the repayment status in August, 2005; ...; PAY_6 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; ...; 8 = payment delay for eight months; 9 = payment delay for nine months and above |
| **BILL_AMT1 - BILL_AMT6** | inputs | float | amount of bill statement; BILL_AMNT1 = amount of bill statement in September, 2005; BILL_AMT2 = amount of bill statement in August, 2005; ...; BILL_AMT6 = amount of bill statement in April, 2005 |
| **PAY_AMT1 - PAY_AMT6** | inputs | float | amount of previous payment; PAY_AMT1 = amount paid in September, 2005; PAY_AMT2 = amount paid in August, 2005; ...; PAY_AMT6 = amount paid in April, 2005 |
| **DELINQ_NEXT**| target | int | whether a customer's next payment is delinquent (late), 1 = late; 0 = on-time |

* **Source of training data**: GWU Blackboard, email `jdoe@gwu.edu` for more information
* **How training data was divided into training and validation data**: 50% training, 25% validation, 25% test
* **Number of rows in training and validation data**:
  * Training rows: 15,000
  * Validation rows: 7,500

### Test Data
* **Source of test data**: GWU Blackboard, email `jdoe@gwu.edu` for more information
* **Number of rows in test data**: 7,500
* **State any differences in columns between training and test data**: Both training and test data came from the same file which are selected randomly.

### Model details
* **Columns used as inputs in the final model**: 'LIMIT_BAL',
       'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1',
       'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
* **Column(s) used as target(s) in the final model**: 'DELINQ_NEXT'
* **Type of model**: Decision Tree 
* **Software used to implement the model**: Python, scikit-learn
* **Version of the modeling software**: Python version: 3.10.7, scikit-learn version: 1.1.2
* **Hyperparameters or other settings of your model**: 
```
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=6, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=12345, splitter='best')
```
### Quantitative Analysis

We used Area Under the ROC Curve (AUC) to assess the performance of the model. The ROC curve is defined as the Receiver Operating Characteristic curve. It shows the performance of a classification model at all classification thresholds. The ROC curve plots two variables: 
* True Positive Rate 
* False Positive Rate

AUC gives us the area under the ROC curve which is the area from FP Rate = 0 to FP Rate = 1. A model with an AUC of 0.0 is 100% wrong while a model with an AUC of 1.0 is 100% correct. Real-world models are in between theses two values. 

We also use Adverse Impact Ratios (AIR) to see analyze the impact of the classification model. Adverse impact is the negative effect an unfair and biased selection procedure has on a protected class. It occurs when a protected group is discriminated against during a selection process, like approving loans for the bank. If the selection rate for a certain group is less than 80% of that of the group with the highest selection rate, there is an adverse impact on that group. 

Based on our final model, the computed AUCs and AIR are shown in the table below with varying depth of Decision Tree Classifier.

| Depth | Training AUC | Validation AUC | 5-Fold SD | Hispanic-to-White-AIR |
| ----- | ------------ | -------------- | --------- | --------------------- |
| 1 | 0.645748 |	0.643880|	0.009275|	0.894148|
| 2|	0.699912|	0.687752	|0.012626|	0.850871|
|3	|0.742968|	0.729490|	0.017375|	0.799546|
|4	|0.757178|	0.741696|	0.017079|	0.792435|
|5	|0.769331|	0.742480|	0.019886|	0.829336|
|6	|0.783722|	0.749610|	0.017665|	0.833205|
|7	|0.795777|	0.742115|	0.022466|	0.835886|
|8	|0.807291|	0.739990|	0.015567|	0.811300|
|9	|0.822913|	0.727224|	0.012042|	0.811561|
|10	|0.838052|	0.720562|	0.013855|	0.803621|
|11	|0.855168|	0.709864|	0.010405|	0.837806|
|12	|0.874251|	0.688074|	0.008073|	0.844889| 

The data in the table above is plotted below to see the variation of the metrics on different values of Decision Tree Classifier depths.

![Iteration Plot](data_plot.png)

We also analyzed the correlation between pairs of columns in the raw data. The correlation heatmap is shown in the image below. 

![Correlation Heatmap](corr_heatmap.png)

We also computed the data histograms for each parameter so we can see the distribution of values for each parameter. We can see from the figure that most of the parameters are skewed to the lower value of the range. The skew could affect the accuracy of the measure of central tendency that we used. 

![Data Histograms](data_histograms.png)

Our best model is the Decision Tree with depth of 7. This is the depth with the maximum Training AUC. At this point, the Validation AUC is 0.742115 and the Hispanic-to-White AIR is 0.835886. This means that using this model will not put the Hispanic group in adverse situation. The resulting tree is shown below.

![Decision Classifier Tree Plot](tree_plot.png)

Based on the results of the training data, we also analyzed the features as to its effect on the dependent varaible. The plot below shows a comparison of the importance of variables. The more important a variable is, the more it affects the result. We can see that the variable PAY_0 has the largest influence on the program. The first payment seems to be the major contributor. The least important variable is the PAY_5 variable. This means that the 6th payment has the least importance in the approval of loan.

![Variable Importance](variable_importance.png)

We have seen in the graph above the effect of credit history on the approval of loan. But it is known in real-life situation that demographics also affects loan approval. At this part, we want to check if there is any bias on the approval among the demographic variables. We used the confusion matrix to evaluate the biases. The tables are shown below.

#### **Confusion Matrix by Hispanic Race**
| | actual: 1 | actual: 0 |
|-|-----------|-----------|
| **predicted: 1** | 447 | 387 |
| **predicted: 0** | 139 | 501 | 

#### **Confusion Matrix by Black Race**
| | actual: 1 | actual: 0 |
|-|-----------|-----------|
| **predicted: 1** | 449 | 348 |
| **predicted: 0** | 157 | 537 | 

#### **Confusion Matrix by White Race**
| | actual: 1 | actual: 0 |
|-|-----------|-----------|
| **predicted: 1** | 176 | 813 |
| **predicted: 0** | 72 | 1228 | 

#### **Confusion Matrix by Asian Race**
| | actual: 1 | actual: 0 |
|-|-----------|-----------|
| **predicted: 1** | 186 | 784 |
| **predicted: 0** | 59 | 1217 | 

We then compare the acceptance rates by race by computing the AIR between the two races. The results are listed in the tables below. Based on the AIR, the black and asian races are not in adverse risk since their acceptance rate is more than 80% that of white. However, the hispanic race is at adverse risk due to its AIR below 80%. We can see here that the White race has the highest acceptance rate and other races are lower than that. 

#### **Comparison of Acceptance Rates by Race**
| Race | Acceptance Rate | AIR (compared to White) |
|------|-----------------|-------------------------|
| White | 0.568 | 1.00 |
| Hispanic | 0.434 | 0.76 |
| Black | 0.465 | 0.82 |
| Asian | 0.568 | 1.00 |

Besides race, we also analyze the AIR between female and male sexes. The confusion matrix showing the acceptance rate is shown in the tables below.

#### **Confusion Matrix by Male**
| | actual: 1 | actual: 0 |
|-|-----------|-----------|
| **predicted: 1** | 546 | 905 |
| **predicted: 0** | 179 | 1292 | 

#### **Confusion Matrix by Female**
| | actual: 1 | actual: 0 |
|-|-----------|-----------|
| **predicted: 1** | 712 | 1427 |
| **predicted: 0** | 248 | 2191 | 

Looking at the data by sexes, we can see almost similar acceptance rate between the two sexes. Hence, we can see that there is little difference between the two sexes in terms of acceptance rate.

#### **Comparison of Acceptance Rates by Sex**
| Sex | Acceptance Rate | AIR (compared to Male) |
|------|-----------------|-------------------------|
| Male | 0.503 | 1.00 |
| Female | 0.533 | 1.06 |

From the analysis above, we have to adjust the model not to put the Hispanic race at risk. Right now, we have AIR of 0.76 which is less than 0.80. From here, we adjusted the cut-off to fix this issue. After several tries, we arrived at a cut-off of 0.18. We recomputed the confusion matrices and the acceptance rates. The results are shown below.

With the adjusted cut-off, we were able to increase the acceptance rates across all races and sexes. From the table below, we can see that the AIR of Hispanic race increased from 0.76 to 0.83. The AIR of Black race increased from 0.82 to 0.85. From the AIR between female and male, we can see here that the acceptance rates are close to each other. 

#### **Comparison of Revised Acceptance Rates by Race**
| Race | Revised Acceptance Rate | AIR (compared to White) |
|------|-------------------------|-------------------------|
| White | 0.735 | 1.00 |
| Hispanic | 0.613 | 0.83 |
| Black | 0.626 | 0.85 |
| Asian | 0.739 | 1.00 |

#### **Comparison of Revised Acceptance Rates by Sex**
| Sex | Revised Acceptance Rate | AIR (compared to Male) |
|-----|-------------------------|-------------------------|
| Male | 0.682 | 1.00 |
| Female | 0.696 | 1.02 |

With the updated cut-off, we recomputed the AUC for various tree depths. We calculated from 1 to 12 tree depth. The AUC are shown below. Here we can see that there is a gradual increase in the training AUC. This is due to overfitting of data. We must look at the validation AUC since it involves data points that are not included in the training. From the iteration plot below, we can see that the validation AUC drops as we increase the depth to more than 6. As the model is overfitted, it gets less accurate to other parts of the data. 

![Iteration Plot Rev](data_plot_rev1.png)

### Ethical Considerations

#### Potential Negative Impacts
*Math or Software Problems:* 
Using this model is based on running through a small dataset. Mathematically, this means that the data is not reliable since it is not representative of the population. It requires further testing to larger dataset so we can verify that it maintains a reasonable AUC and AIR levels. In addition to that, we have used scikit learn software as a model which is relatively stable. Scikit learn may not be fast enough if we train the model using a larger dataset. Hence, we can use other modelling software with larger data capacity and faster execution time. 

*Real-world risks:* 
Due to concerns on reliability of the model, we must be careful in using this in the actual world. Disapproving loans to credit worthy invdividuals due to a mistake on this model can be a matter of losing a home or business for that individual. This could pose real life effects that.

#### Potential Uncertainties
*Math or Software Problems:*
Due to the limited dataset used in our model, the prediction can have a high uncertainty. Since we are using third-party software to run the model, there might be bugs and issues in it that will affect the execution of the model. 

*Real-world risks:* 
Real-world risks are there in the use of this model. If banks will use this model as basis of their decision, this will affect the life of people. 