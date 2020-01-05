#############################
# Iris-Classification Project
#############################

# Install Necessary Packages
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

# Load dataset from csv file
data <- read.csv("iris.csv", header=TRUE)

### Dataset summary

# Dataset dimensions
dim(data)

# View headers and types of columns
sapply(data, class)

# List of Species class levels
levels(data$Species)

# Statistcal summary of dataset
summary(data)

# Distribution of Species by frequency and percentage
percentage <- prop.table(table(data$Species)) * 100
cbind(freq=table(data$Species), percentage=percentage)

# Split dataset into x and y, y being class labels
x <- data[,2:5]
y <- data[,6]

# Boxplot for each attribute
par(mfrow=c(1,4))
for(i in 1:4) {
  boxplot(x[,i], main=names(x)[i])
}

# Barplot showing frequency of each class
plot(y)

# Multivariate plots
# Box and whisker plots by class for each attribute
featurePlot(x=x, y=y, plot="box")

# Density plots by class for each attribute
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)


### Machine Learning Model Building

#Split the dataset into training and test set using createDataPartition(), 80% of data as training set and 20% of data as test set
test_index <- createDataPartition(data$Species, p = 0.8, list = FALSE)
train <- data[test_index,]
test <- data[-test_index,]

# Algorithms will be assessed using 10-fold crossvalidation, setup here
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Linear Discriminant Analysis
set.seed(1)
fit.lda <- train(Species~., data=data, method="lda", metric=metric, trControl=control)
predictions.lda <- predict(fit.lda,test)

# Decision Tree
set.seed(1)
fit.rpart <- train(Species~., data=data, method="rpart", metric=metric, trControl=control)
predictions.rpart <- predict(fit.rpart,test)

# k-Nearest Neighbors
set.seed(1)
fit.knn <- train(Species~., data=data, method="knn", metric=metric, trControl=control)
predictions.knn <- predict(fit.knn,test)

# Support Vector Machines
set.seed(1)
fit.svm <- train(Species~., data=data, method="svmRadial", metric=metric, trControl=control)
predictions.svm <- predict(fit.svm,test)

#Random Forest
set.seed(1)
fit.rf <- train(Species~., data=data, method="rf", metric=metric, trControl=control)
predictions.rf <- predict(fit.rf,test)

### Summarize model accuracies
# Summary of Accuracy and Kappa of different models
results <- resamples(list(lda=fit.lda, cart=fit.rpart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

dotplot(results)

# Evaluate confustion matrix of the models' predictions on test data
confusionMatrix(predictions.lda, test$Species)
confusionMatrix(predictions.rpart, test$Species)
confusionMatrix(predictions.knn, test$Species)
confusionMatrix(predictions.svm, test$Species)
confusionMatrix(predictions.rf, test$Species)

# Create a table to summarise the accuracy of different models
## lda
cm <- confusionMatrix(predictions.lda, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- tibble(Model = "lda", Accuracy = overall.accuracy)
## rpart
cm <- confusionMatrix(predictions.rpart, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="rpart",
                            Accuracy = overall.accuracy))
## knn
cm <- confusionMatrix(predictions.knn, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="knn",
                            Accuracy = overall.accuracy))
## svm
cm <- confusionMatrix(predictions.svm, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="svm",
                            Accuracy = overall.accuracy))
## rf
cm <- confusionMatrix(predictions.rf, test$Species)
overall <- cm$overall
overall.accuracy <- overall['Accuracy'] 

Summary <- bind_rows(Summary, 
                     tibble(Model="rf",
                            Accuracy = overall.accuracy))

# Model Accuracy Summary
print(Summary)

