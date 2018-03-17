 B565: Data Mining
 Homework Assignment 2
 Author : Akshay Naik
*****************************************************************************************

   
  Problem:
  K-means with Elkan's acceleration 

  Program Name:
  K_means_clustering_using_Elkan-acceperation.py

  Programming language:
  Python 2

  Packages used:
  numpy, pandas, sys, copy 
  
  Execution Details:
  This program can be run on hulk server using below command

  'python K_means_clustering_using_Elkan-acceperation.py k'

  Parameter Description and values (Note: All Parameters are manditory)

  k : Number of clusters to be considered {3,20,100, any other number}

  Some sample commands:
  python K_means_clustering_using_Elkan-acceperation.py 3
  python K_means_clustering_using_Elkan-acceperation.py 20
  python K_means_clustering_using_Elkan-acceperation.py 100

  Details:
  I have used Birch dataset for this problem. The code is not vectorized
  so it will take alot of time to execute for higher value of k but it takes
  around 2-4 minutes for k = 3        
---------------------------------------------------------------	

  Problem 3-c,3-d:
  K-means  

  Program Name:
  K_means_clustering.py

  Programming language:
  Python 2

  Packages used:
  numpy, pandas, sys, copy 
  
  Execution Details:
  This program can be run on hulk server using below command

  'python K_means_clustering.py dataset distance_function'

  Parameter Description and values (Note: All Parameters are manditory)

  dataset : The name of the dataset to be used which will have values {iris, balance_scale, glass}

  distance_function : Specify which distance function to be used. It will have values
                      {cityblock, cosine, fn_1, fn_2, -, eucledian}.
                      By default it will consider eucledian distance as distance function  

  Some sample commands:
  python K_means_clustering.py iris cosine
  python K_means_clustering.py glass - 
  python K_means_clustering.py balance_scale fn_1

  Details:
  This code uses url to get the data so just the name of the dataset 
  to be used is enough to run the code as url's for the dataset mentioned
  are included in the code.           