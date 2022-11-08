# ActiveLearning

**Project:** Active Learning Algorithms and Tutorial

**Description:**  

Two Active Learning algorithms were developed in python using simple packages. A Random active learning algorithm that samples the unlabeled data randomly at each iteration step. And an Entropy-base Active learning algorithm that samples the unlabeled data with the highest uncertainty at each iteration step. The default number of iterations (N) is 50 and the number of samples taken from the unlabeled set (k) was 10, at each iteration. The Scikit-Learn Logistic Regression model with a `liblinear` solver and max iterations set to 300 was used to fit the training data at each step and make predictions on the test data to calculate accuracy. 

The code file contains the function for importing multiple MATLAB datasets from a file and the two functions that implement the Active Learning algorithms. The datasets used in the `Jupyter notebook` tutorial are: MindReading and MMI data sets. Both data sets are in MATLAB format and were converted to a pandas DataFrame format before running the algorithms. 

**Dependencies:**
Libraries needed to run the code and their versions.

os.

pandas - 1.3.5. 

scipy -  1.7.3. 

numpy - 1.20.3. 

sklearn - 1.0.2. 

matplotlib - 3.5.0. 
