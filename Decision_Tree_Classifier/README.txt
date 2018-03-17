  Problem 6 :
  Decision Tree

  Program Name :
  Decision_Tree_Classifier.py

  Programming Language :
  Python 2

  Packages used:
  pandas, numpy, copy, sys

  Execution Details:
  This program can be run in hulk server using the below command:

  'python Decision_Tree_Classifier.py dataset_name which_measure link_of_other_dataset'
 
  Parameter Descriptions and Values (Note: All Parameters are mandatory):

  dataset_name : Specify the dataset name which are used by me.
                 Different values it might have are {iris, haberman, cancer, -}.
                 specify '-' when other dataset is been used to test the code.  

  which_measure : Specifies whether to use gini or information gain.
                  Different values it might have are {information_gain, gini, -}.
                  By default the code will consider gini as impurity measure.

  link_of_other_dataset : This argument can be used to give link of other dataset which need to used to test the code.

  Some sample commands:
  python Decision_Tree_Classifier.py iris information_gain -
  python Decision_Tree_Classifier.py - gini https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

  Description:
  I have used link of dataset instead of file as input data.
  The three dataset I have used have their link embedded in the code so the first argument 
  takes care of it. For new dataset, the third argument should be used to specify the link of
  dataset.
  I have assumed that always the last attribute will be the target attribute. In case this is
  not true, I will preprocess the data to bring it in a fromat that my code can easily handle.
 