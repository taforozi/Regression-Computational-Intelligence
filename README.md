# Regression-Computational-Intelligence
The purpose of this assignment is to investigate the ability of **TSK** (Takagi-Sugeno-Tang) models to fit multivariable, nonlinear functions. Especially, using two datasets from the [UCI Repository](https://archive.ics.uci.edu/ml/index.php) and fuzzy neural models, we are trying to estimate the target attribute from the available data. 

## Part 1
The first dataset is used for a simple investigation of the training and evaluation process of such models, as well as illustrating manners of analyzing and interpreting the results. The models that are examined are four and they differ from each other in the number of membership functions (2 or 3) and the type of their output (Singleton or Polynomial). In this case, the small size of the dataset allows us to use the **Grid Partition** method for the input space division.

###### dataset: [Airfoil Self-Noise dataset](https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise)

## Part 2
The second and more complicated dataset is used for a more complete modelling process, which involves, among others, **preprocessing steps** such as **feature selection** and methods for optimizing models through **cross validation**. Due to the large size of the dataset, problems such as rule explosion may appear. In order to avoid that, firstly, we deploy another method called **Subtractive Clustering (SC)** for the input partition and it is also necessary that we decrease dataset's dimensionality by choosing the most significant features and reject the less useful ones. After that, we apply **Grid Search** and **5-fold Cross Validation** to find the best combination of the number of features and cluster radius, which leads to the minimum validation error. Using the results that arise from that procedure, we train the final model and we evaluate it according to the **MSE, NMSE, NDEI and R^2** metrics.

###### dataset: [Superconductivty dataset](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data)

## Code
Τhe files have been created using the **MATLAB R2018a** version. If a different version is used, you may need to customize some commands.

## Contact
If there are any questions, feel free to [contact me](mailto:thomi199822@gmail.com?subject=[GitHub]%20Source%20Han%20Sans). 
