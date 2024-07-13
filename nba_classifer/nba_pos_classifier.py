import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Step-1: Creating train and test data sets and using linear svm to train the model
# Read data from CSV file
nba = pd.read_csv('nba2021.csv')
# Filter data to only include players who played more than 15 minutes
nba = nba[nba['MP'] > 15]
# Extract the target variable
output_data = nba['Pos']
# Extract the training features droping the irrelvant columns
train_data = nba.drop(['Player','Tm','FT%','Pos','G','GS','PTS','MP','TOV','eFG%'],axis=1)
# Fit and transform the training features using MinMaxScaler
scaler = MinMaxScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    scaled_train, output_data, stratify=output_data, test_size=0.25)
# Create a LinearSVC model with balanced class weights and a maximum of 1500 iterations
lsvm = LinearSVC(class_weight='balanced', max_iter=1500)
# Fit the model on the training data
lsvm.fit(x_train, y_train)


# Step-2: Accuracy of the liner svm model which is trained in step-2
# Use the trained model to predict the class labels of the test features
pred = lsvm.predict(x_test)
# Compute the accuracy of the model on the test set
avg_accuracy = (pred == y_test).mean()
# Print the test set score with three decimal places
print(f"Accuracy of the Model : {avg_accuracy:.4f}")

# Step- 3: confusion matrix
# Use the trained model to predict the class labels of the test features
pred = lsvm.predict(x_test)
# Compute the confusion matrix
conf_mat = confusion_matrix(y_test, pred, labels=lsvm.classes_)
# Create a DataFrame to display the confusion matrix
conf_df = pd.DataFrame(conf_mat, index=lsvm.classes_, columns=lsvm.classes_)
# Add row and column totals to the confusion matrix
conf_df['Total'] = conf_df.sum(axis=1)
conf_df.loc['Total'] = conf_df.sum()
# Print the confusion matrix
print("Confusion matrix:")
print(conf_df)

# Step-4: 10-fold stratified cross-validation.
# Define a StratifiedKFold object with 10 folds and shuffling
sKFold = StratifiedKFold(n_splits=10, shuffle=True)
# Initialize a list to store the fold accuracies
accuracy_list = []
# Iterate over the folds
for f, (train, test) in enumerate(sKFold.split(scaled_train, output_data)):
    # Train the model on the training data for this fold
    lsvm.fit(scaled_train.iloc[train, :], output_data.iloc[train])
    # Evaluate the model on the test data for this fold
    acy = lsvm.score(scaled_train.iloc[test, :], output_data.iloc[test])
    # Append the fold accuracy to the list
    accuracy_list.append(acy)
    # Print the fold number and accuracy
    print('Fold: %2d, Accuracy: %.4f' % (f + 1, acy))

# Step-5: Accuracy of each fold and average accuracy 10-folds
# Compute the average accuracy across all folds
mean_accuracy = np.mean(accuracy_list)
# Print the average accuracy
print('Cross-Validation Average Accuracy: %.3f' % mean_accuracy)
