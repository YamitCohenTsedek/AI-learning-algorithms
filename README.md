## Implementation of ID3, KNN and Naive Base learning algorithms
<br />

### Dataset for the learning process
File dataset.txt contains the dataset examples for the learning process, when the first line of this file includes the field names (i.e., the attributes of the data).<br />
The possible values of each attribute are the values that appear in the attribute column in the dataset file.<br /><br />

### ID3 algorithm
ID3 (Iterative Dichotomiser 3) is an algorithm used to generate a decision tree from a dataset.<br />
In our implementation, the decision tree built from the algorithm is printed to a file named tree.txt in the following format:<br />
<attribute_name>=<attribute_value><br />
\<tab>|<attribute_name>=<attribute_value>:class<br />
  
The printing of the values of the attributes is in alphabetical order, for example:<br />
age = child<br />
| pclass = crew: yes<br />
| pclass = 1st: yes<br />
| pclass = 2nd: yes<br />
| pclass = 3rd: no<br />
<br />

### KNN algorithm
KNN (k-nearest neighbors) algorithm is a non-parametric method used for classification and regression.<br />
Our implementation of KNN algorithm is for a classification problem, when K=5. Our distance heuristic is Hamming distance.<br />
Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different.<br />
<br />

### Naive Base algorithm
Naive Bayes is a probabilistic machine learning algorithm based on the Bayes Theorem.<br />
<br />

### Accuracy
After building the model for each of the algorithms, the accuracy of the predictions is printed (with a two decimal place accuracy) to a file named accuracy.txt in the following format:<br />
<DT_accuracy>tab<KNN_accuracy>tab<NaiveBase_accuracy><br />
<br />

### Test phase
For the test phase the program gets train.txt & test.txt files. It builds the three classifiers from the train.txt dataset and prints the results to a file named output.txt in the following format:<br />
The content of tree.txt file.<br />
\<space line><br /> 
The content of accuracy.txt file when the accuracy is for the test.txt file.<br />
