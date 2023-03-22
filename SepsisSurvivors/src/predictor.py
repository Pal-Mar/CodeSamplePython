"""_summary_ Python Code Sample - Predicting outcome of sepsis (body's extreme response to an infection) 
using a K-nearest neighbors classifier

Please see also original article at https://www.nature.com/articles/s41598-020-73558-3 
[Chicco, D., Jurman, G. Survival prediction of patients with sepsis from age, sex, and septic episode 
number alone. Sci Rep 10, 17156 (2020). https://doi.org/10.1038/s41598-020-73558-3]

Data sets and descriptions of data available from UCI Machine Learning repository at 
[https://archive.ics.uci.edu/ml/datasets/Sepsis+survival+minimal+clinical+records#]

Data sets distributed with following Licence: CC BY 4.0: freedom to share and adapt the material

"""

from sklearn.neighbors import KNeighborsClassifier
import pandas

def main():
    """_summary_ This function demonstrates use of K nearest neighbours classifier on sample data set, 
    here data set is sepsis survival data retrieved from UCI Machine Learning repository. 
    Features used in data: [[age_years, sex_0male_1female, episode_number, hospital_outcome_1alive_0dead]]
    """

    selection, path = selectInputFile()
    patients = pandas.read_csv(path) # file read into Pandas DataFrame object

    outcomeColumn = "hospital_outcome_1alive_0dead"
    a = patients.drop(outcomeColumn, axis=1) # this column contains actual outcome, thus dropped
    b = patients[outcomeColumn] # this column contains actual outcome
    from sklearn.model_selection import train_test_split
    aTrain, atest, bTrain, bTest = train_test_split(a, b, test_size=0.2, random_state=0)
  
    kNearestN = KNeighborsClassifier(n_neighbors=1)
    kNearestN.fit(aTrain, bTrain)

    from datetime import datetime
    stamp = int(datetime.timestamp(datetime.now())) # stamp for use in testing output filename

    selections = ["", "Small", "Medium", "Large"]

    # This testing is for a larger project to do testing on parameters and k-size
    # Here tests are simply run for all samples of same data set used to generate classifier
    runTests(patients, stamp, kNearestN, selections[selection])


def runTests(patients: pandas.DataFrame, stamp: int, kNearestN: KNeighborsClassifier, filetype: str):
    filename = f"results{filetype}Set{str(stamp)}.txt" #file saved in results[dataSet][stamp].txt
    testDF = patients.copy()
    testDF = testDF.reset_index()  # making sure indexes pair with number of rows
    incorrectPredictions = 0

    # run test on ALL rows of input  
    for patient in testDF.itertuples(): #used itertuples for Code Sample purposes only - other structures faster for large sets.
        testData = [patient[2], patient[3], patient[4]]   

        sex, outcome, expected = parseRow(patient)
        caseDescription = f"{patient[2]} year old {sex}, {patient[4]} episode(s), patient {outcome}"
        # e.g. {47} year old female, {2} episode(s), patient {survived}

        incorrectPredictions += testingPredictions(filename, kNearestN, testData, expected, caseDescription)
    
    print("Incorrect predictions detected:", incorrectPredictions)
    with open(filename, "a") as newFile:
        line = f"Incorrect predictions: {incorrectPredictions} out of {len(testDF.index)}"
        newFile.write(line)   


import warnings
import numpy
def testingPredictions(filename: str, knn: KNeighborsClassifier, input: list, expected, description: str):
    # this tests the prediction with inputs to the chosen model - here K nearest neighbours 
    # features used in data: [[age_years, sex_0male_1female, episode_number, hospital_outcome_1alive_0dead]]

    features = numpy.array([input])  # read inputdata into array using numpy
    warnings.filterwarnings("ignore")
    prediction = knn.predict(features)   # take inputdata and generate prediction

    # here prediction compared to outcome - later on written to file with explanation of case if prediction false
    result = f"{input} - Prediction: {prediction[0]} - Expected: {expected}"

    incorrect = 0
    if str(prediction[0]) != expected:
        result += " - " + description + "\n"
        incorrect = 1
    else:
        result += "\n"
    
    with open(filename, "a") as newFile:  #  file saved in results[dataSet][stamp].txt
        newFile.write(result)   
    return incorrect    


def selectInputFile():
    while(True):
        selection = input("Which file to use? 1 - small set, 2 - medium set, 3 - large set")
        if selection == "1":
            path = "SMALL_sepsis_survival_study.csv"
            return (selection, path)
        elif selection == "2":
            path = "MEDIUM_sepsis_survival_study.csv"
            return (selection, path)
        elif selection == "3":
            path = "LARGE_sepsis_survival_study.csv"
            return (selection, path)
        else:
            print("Incorrect selection, please choose 1, 2 or 3!")


def parseRow(patient):
    if patient[3] == 0:
        sex = "male"
    else:
        sex = "female"
    if patient[5] == 0:
        outcome = "died"
    else: 
        outcome = "survived"
    expected = str(patient[5])    
    return (sex, outcome, expected)


main()