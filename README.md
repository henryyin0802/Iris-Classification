# Iris-Classification
Iris Species Classification Machine Learning Project

# Executive Summary
The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

1.Id - unique ID of the samples
2.SepalLengthCm - Length of the sepal (in cm)
3.SepalWidthCm - Width of the sepal (in cm)
4.PetalLengthCm - Length of the petal (in cm)
5.PetalWidthCm - Width of the petal (in cm)
6.Species - Species name

The aim of the project is to create a machine learning algorithm to predict the iris species correctly based on the given attributes.

# Machine Learning Methods
### Install Necessary Packages
```{r Install Necessary Packages}
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
```

### Load dataset from csv file
```{r Load dataset from csv file}
data <- read.csv("iris.csv", header=TRUE)
```

### Dataset summary
#### Dataset dimensions
```{r Dataset dimensions}
dim(data)
```
####View headers and types of columns
```{r View headers and types of columns}
sapply(data, class)
```
####List of Species class levels
```{r List of Species class levels}
levels(data$Species)
```
####Statistcal summary of dataset
```{r Statistcal summary of dataset}
summary(data)
```
####Distribution of Species by frequency and percentage
```{r Distribution of Species by frequency and percentage}
percentage <- prop.table(table(data$Species)) * 100
cbind(freq=table(data$Species), percentage=percentage)
```
####Split dataset into x and y, y being class labels
```{r Split dataset into x and y, y being class labels}
x <- data[,2:5]
y <- data[,6]
```
####Boxplot for each attribute
```{r Boxplot for each attribute, echo=FALSE}
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(x)[i])
}
```
####Barplot showing frequency of each class
```{r Barplot showing frequency of each class, echo=FALSE}
plot(y)
```
####Box and whisker plots by class for each attribute
```{r Box and whisker plots by class for each attribute, echo=FALSE}
featurePlot(x=x, y=y, plot="box")
```
####Density plots by class for each attribute
```{r Density plots by class for each attribute, echo=FALSE}
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)
```

### Machine Learning Model Building
####Split the dataset into training and test set using createDataPartition(), 80% of data as training set and 20% of data as test set
```{r Split dataset into training and test set}
test_index <- createDataPartition(data$Species, p = 0.8, list = FALSE)
train <- data[test_index,]
test <- data[-test_index,]
```
####Algorithms will be assessed using 10-fold crossvalidation, setup here
```{r 10-fold crossvalidation setup}
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
```
####5 machine learning models are introduced and respective accuracy of the prediction on test set are compared
Linear Discriminant Analysis
```{r lda model building}
set.seed(1)
fit.lda <- train(Species~., data=data, method="lda", metric=metric, trControl=control)
predictions.lda <- predict(fit.lda,test)
```
Decision Tree
```{r rpart model building}
set.seed(1)
fit.rpart <- train(Species~., data=data, method="rpart", metric=metric, trControl=control)
predictions.rpart <- predict(fit.rpart,test)
```
k-Nearest Neighbors
```{r knn model building}
set.seed(1)
fit.knn <- train(Species~., data=data, method="knn", metric=metric, trControl=control)
predictions.knn <- predict(fit.knn,test)
```
Support Vector Machines
```{r svm model building}
set.seed(1)
fit.svm <- train(Species~., data=data, method="svmRadial", metric=metric, trControl=control)
predictions.svm <- predict(fit.svm,test)
```
Random Forest
```{r rf model building}
set.seed(1)
fit.rf <- train(Species~., data=data, method="rf", metric=metric, trControl=control)
predictions.rf <- predict(fit.rf,test)
```

#Results
### Summarize model accuracies
####Summary of Accuracy and Kappa of different models
```{r accuracy and kappa of models}
results <- resamples(list(lda=fit.lda, cart=fit.rpart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

dotplot(results)
```
####Evaluate confustion matrix of the models' predictions on test data
```{r confusion matrix}
confusionMatrix(predictions.lda, test$Species)
confusionMatrix(predictions.rpart, test$Species)
confusionMatrix(predictions.knn, test$Species)
confusionMatrix(predictions.svm, test$Species)
confusionMatrix(predictions.rf, test$Species)
```
### Create a table to summarise the accuracy of different models
Linear Discriminant Analysis
```{r models summary table - lda}
cm <- confusionMatrix(predictions.lda, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- tibble(Model = "lda", Accuracy = overall.accuracy)
```
Decision Tree
```{r models summary table - rpart}
cm <- confusionMatrix(predictions.rpart, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="rpart",
                            Accuracy = overall.accuracy))
```
k-Nearest Neighbors
```{r models summary table - knn}
cm <- confusionMatrix(predictions.knn, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="knn",
                            Accuracy = overall.accuracy))
```
Support Vector Machines
```{r models summary table - svm}
cm <- confusionMatrix(predictions.svm, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="svm",
                            Accuracy = overall.accuracy))
```
Random Forest
```{r models summary table - rf}
cm <- confusionMatrix(predictions.rf, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="rf",
                            Accuracy = overall.accuracy))
```
### Print summary table of models' accuracy
```{r Print summary table}
print(Summary)
```

#Conclusion
In the project, 5 models are introduced: Linear Discriminant Analysis (lda), Decision Tree (rpart), k-Nearest Neighbors (knn), Support Vector Machines (svm), Random Forest (rf). Algorithms are built based on train set data and are applied to test set for prediction. Accuracy of predictions of different models is summarised in the table. Based on the result, all models give 100% accuracy. 
