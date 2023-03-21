# Predictor - Python Code Sample

Predicting outcome of sepsis (body's extreme response to an infection) using K-nearest neighbors classifier as an example of using a machine learning for prediction purposes. This code sample was used to test the classifier with various parameters and sample sizes.

Data originally from research by Chicco, D., Jurman G. and published as ['Survival prediction of patients with sepsis from age, sex, and septic episode number alone' in Sci Rep 10(2020)](https://doi.org/10.1038/s41598-020-73558-3). 

**note**
Training data sets used are distributed by original authors with following Licence: 'CC BY 4.0: freedom to share and adapt the material'

# Program implementation
This version demonstrates use of K nearest neighbors classifier on sample data sets, here data sets are of survival data following a sepsis diagnosis. The chosen classifier implementation is [sklearn.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) from [scikit-learn](https://scikit-learn.org/stable/), which implements the k-nearest neighbors vote. See more on [Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification).

The original data (.csv) is read into a [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) and split into two structures: data for generating prediction (3 columns), data of outcome (1 column).

The neighbors classifier is given a parameter for how many neighbors to include, this version tested using k-neighbours up to 30. Most accurate predictions were derived with k=20 for both Medium and Large data sets, with Small data set showing best results with smaller k-size. From KneighbourClassifier [documentation](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification): 'The optimal choice of the value is highly data-dependent: in general a larger suppresses the effects of noise, but makes the classification boundaries less distinct.'

Testing was also done on test-size of [sklearn train_test_split](https://www.sharpsightlabs.com/blog/scikit-train_test_split/),with 0.2 staying as best result for these data sets.

Best performance for Small data set with 137 samples was k=1 - 10 incorrect predictions - 0.92 accuracy
Best performance for Medium data set with 19051 samples was at k=20 - 3606 incorrect predictions - 0.81 accuracy
Best performance for Large data set with 110204 samples was k=20 - 8105 incorrec predictions - 0.92 accuracy

Please note, results and incorrect predictions were NOT analyzed for bias for age or sex, results were based purely on number of incorrect predictions.

# Data
This version comes with 3 data sets included (.csv), here named Small, Medium and Large. Data sets can be found in data-folder as well as downloaded directly from [UCI ML repository](https://archive.ics.uci.edu/ml/datasets/Sepsis+survival+minimal+clinical+records#). 

Three sample output result-files are also appended, with data from a run with n/k = 1 and test_size = 0.2. 

Features in data: age_years, sex_0male_1female, episode_number, hospital_outcome_1alive_0dead. 
Columns contain information as integers about age of patient at time of hospitalization, sex(binary), number of episode with this condition as well as hospitalization outcome (survival).


# Credits

Data available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Sepsis+survival+minimal+clinical+records#) Data sets distributed by original authors with following Licence: 'CC BY 4.0: freedom to share and adapt the material'

Classifier from [scikit-learn](https://scikit-learn.org/stable/index.html)

Data collected for original research in several phases: Please see [original article](https://www.nature.com/articles/s41598-020-73558-3) 