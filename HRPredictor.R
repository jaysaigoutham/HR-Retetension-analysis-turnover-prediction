
############################################
#Importing Libs
############################################
library(rpart)
library(caret) # for identifying correlation
library(corrplot) # for correlation matrix
library(logger)
library(dplyr)
library(gbm)
library(ROCR)
library(tree) # For Decision Tree
library(e1071)
library(ggplot2)
library(caret)
library(rpivotTable)
library(NeuralNetTools) # For Neural Network tools
library(nnet) # For Neural Network
library(lattice) # For Neural Network
library(rattle)  # graphical interface for data science in R
library(randomForest) # For Random Forest (RF)
library(ROSE) # For Evaluation meathods
library(dplyr) # for data manipulation
library(caret) # for model-building
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculations
library(caretEnsemble) # For ensemble meathods
library(mlbench) # For ensemble meathods
library(ada) #For Adaboost Alogorithm
library(glmnet)
library(DMwR) # For SMOTE
library(Metrics)
library(htmltools)
library(shiny)
library(DT)
library(bslib) # For theme
library(shinythemes) # For theme
library(here) #Fro loading image
############################################
#Utility Functions
############################################

# Function 1 - check for internet conntivity
havingIP <- function() {
  if (.Platform$OS.type == "windows") {
    ipmessage <- system("ipconfig", intern = TRUE)
  } else {
    ipmessage <- system("ifconfig", intern = TRUE)
  }
  validIP <- "((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)[.]){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
  any(grep(validIP, ipmessage))
}
havingIP()


#Function 2 - Normalisation
min_max <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

############################################
#Importing Data
############################################
dataset <- read.csv("HRIS.csv")

number_of_records <- nrow(dataset)
number_of_columns <- ncol(dataset)
print(number_of_records)


############################################
#Dataset validations
############################################

#1.check if dataset is empty, if then exit then and there
if(number_of_records == 0)
{
  stop("Dataset invalid: No rows detected")
  exit()
}

#2.check for NA values
table(is.na(dataset))

#3.check for duplicated values
sum(duplicated(dataset))


############################################
#Transformation & Feature Engineering
############################################

#Drop columns with same value - EmployeeCount, StandardHours, Over18
drop_var<-names(dataset[, nearZeroVar(dataset)])
drop_var
dataset[,c(drop_var)] <- list(NULL)

#dataset <- dataset %>% select(-EmployeeNumber)

#scaling data
dataset$DailyRate<-min_max(dataset$DailyRate)
dataset$DistanceFromHome<-min_max(dataset$DistanceFromHome)
dataset$HourlyRate<-min_max(dataset$HourlyRate)
dataset$MonthlyIncome<-min_max(dataset$MonthlyIncome)
dataset$MonthlyRate<-min_max(dataset$MonthlyRate)
dataset$NumCompaniesWorked<-min_max(dataset$NumCompaniesWorked)
dataset$PercentSalaryHike<-min_max(dataset$PercentSalaryHike)
dataset$TotalWorkingYears<-min_max(dataset$TotalWorkingYears)
dataset$YearsAtCompany<-min_max(dataset$YearsAtCompany)
dataset$YearsInCurrentRole<-min_max(dataset$YearsInCurrentRole)
dataset$YearsSinceLastPromotion<-min_max(dataset$YearsSinceLastPromotion)
dataset$YearsWithCurrManager<-min_max(dataset$YearsWithCurrManager)

#Categorical Variables conversion
dataset$Gender<-as.factor(dataset$Gender)
dataset$BusinessTravel<-as.factor(dataset$BusinessTravel)
dataset$Department<-as.factor(dataset$Department)
dataset$Education<-as.factor(dataset$Education)
dataset$EducationField <- as.factor(dataset$EducationField)
dataset$JobRole <- as.factor(dataset$JobRole)
dataset$MaritalStatus<-as.factor(dataset$MaritalStatus)
dataset$OverTime<-as.factor(dataset$OverTime)
dataset$Attrition<-as.factor(dataset$Attrition)

#2.Factor Variable conversion 
conv_fact<- c("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel","TrainingTimesLastYear","WorkLifeBalance")
dataset[, conv_fact] <- lapply((dataset[, conv_fact]), as.factor)


#feature selection -  not all variables are correlated, to select the highlight correlated ones
#TODO : Should i select teh features here it self ?
dt <- rpart( Attrition~.,data=dataset,control=rpart.control(minsplit = 10))
dt$variable.importance

#Test the correlation between numerical attributes
all_numeric_columns <- unlist(lapply(dataset, is.numeric))  
all_numeric_columns
numeric_coumsn_names <- names(dataset[ , all_numeric_columns])
numeric_coumsn_names
corrplot(cor(dataset[,c(numeric_coumsn_names)]),method = "pie")

# Converting outcome variable to numeric
#dataset$Attrition <- ifelse(dataset$Attrition == "Yes",1,0)

# Centering and scaling data ?
#preProcValues <- preProcess(dataset, method = c("center", "scale"))
#dataset <- predict(preProcValues, dataset)

#check data frame dimensions and column data types
set.seed(100)
str(dataset)
glimpse(dataset)
summary(dataset)


############################################
#Visualization
############################################


table(dataset$Attrition)
ggplot(dataset, aes(x=Attrition, fill=Attrition)) + geom_bar()
#Observation : The Attrition classes are imbalanced 5:1 Ratio

#Attrition vs Overtime
table(dataset$OverTime, dataset$Attrition)
ggplot(dataset, aes(OverTime, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")

summary(dataset$MonthlyIncome)
MnthlyIncome <- cut(dataset$MonthlyIncome, 10, include.lowest = TRUE, labels=c(1,2,3,4,5,6,7,8,9,10))
ggplot(dataset, aes(MnthlyIncome, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")

summary(dataset$HourlyRate)
HrlyRate<- cut(dataset$HourlyRate, 7, include.lowest = TRUE)
ggplot(dataset, aes(HrlyRate, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")


############################################
#Data Splitting
############################################

#Splitting data for training & testing
ModelData<-sample_frac(dataset, 0.75)
sid<-as.numeric(rownames(ModelData)) # because rownames() returns character
ValidateData<-dataset[-sid,]

subject_names <- colnames(ModelData)[colnames(ModelData) != "Attrition"]
Subjective <- c(subject_names)
Objective <- c("Attrition")
TrainingData<-ModelData[c(Subjective, Objective)]
TestingData<-ValidateData[c(Subjective, Objective)]

#checking the dimension of data
dim(TrainingData);
dim(TestingData);


### AUC plot function
# fun.aucplot <- function(pred, obs, title){
#   # Run the AUC calculations
#   ROC_perf <- performance(prediction(pred,obs),"tpr","fpr")
#   ROC_sens <- performance(prediction(pred,obs),"sens","spec")
#   ROC_auc <- performance(prediction(pred,obs),"auc")
#   # Spawn a new plot window (Windows OS)
#   graphics.off(); x11(h=6,w=6)
#   # Plot the curve
#   plot(ROC_perf,colorize=T,print.cutoffs.at=seq(0,1,by=0.1),lwd=3,las=1,main=title)
#   abline(a=0,b=1)
#   # Add some statistics to the plot
#   text(1,0.15,labels=paste("AUC = ",round(ROC_auc@y.values[[1]],digits=2),sep=""),adj=1)
# }

############################################
#Imbalance Correction
############################################

#check classes distribution
prop.table(table(TrainingData$Attrition))
table(TrainingData$Attrition)

#up sampling
upsampled_dataset <- ovun.sample(Attrition ~ ., data = TrainingData, method = "over",
                                 N = length(which(TrainingData$Attrition == "No")) * 2, seed = 100)$data
table(upsampled_dataset$Attrition)

#under sampling
undersampled_dataset <- ovun.sample(Attrition ~ ., data = TrainingData, method = "under", N = length(which(TrainingData$Attrition == "Yes")) * 2, seed = 100)$data
table(undersampled_dataset$Attrition)

#hybrid - oversampling + undersampling
#??? Undersampling makes us loose data, Hybrod apprich of under & over sampling can be used
#In this case, the minority class is oversampled with replacement and majority class is undersampled without replacement.
#p refers to the probability of positive class in newly generated sample
hybrid_sampled_dataset <- ovun.sample(Attrition ~ ., data = TrainingData, method = "both", p=0.5, N=1470, seed = 100)$data
table(hybrid_sampled_dataset$Attrition)

#ROSE - synthetic data generation
#Creates a sample of synthetic data by enlarging the features space of minority and majority class examples. 
#ROSE (Random Over Sampling Examples) package helps us to generate artificial data based on sampling methods and smoothed bootstrap approach.
rose_dataset <- ROSE(Attrition ~ ., data = TrainingData, seed = 100)$data
table(rose_dataset$Attrition)

#SMOTE - Oversampling the minority
#dataset$Attrition <- factor(ifelse(dataset$Attrition =="Yes",1,0))
#smote_dataset <- SMOTE(Attrition ~ .,
#                       data = dataset,
#                       perc.over=300,
#                       perc.under=150)
#table(smote_dataset$Attrition)

#Comparison
prop.table(table(TrainingData$Attrition))
unbalanced_chart <- ggplot(TrainingData, aes(x=Attrition, fill=Attrition)) + geom_bar()
print(unbalanced_chart + ggtitle("1. Un-balanced dataset"))

prop.table(table(undersampled_dataset$Attrition))
undersampled_chart <- ggplot(undersampled_dataset, aes(x=Attrition, fill=Attrition)) + geom_bar()
print(undersampled_chart + ggtitle("2. Undersampled dataset"))

prop.table(table(upsampled_dataset$Attrition))
upsampled_chart <- ggplot(upsampled_dataset, aes(x=Attrition, fill=Attrition)) + geom_bar()
print(upsampled_chart + ggtitle("3. Upsampled dataset"))

prop.table(table(rose_dataset$Attrition))
rose_chart <- ggplot(rose_dataset, aes(x=Attrition, fill=Attrition)) + geom_bar()
print(rose_chart + ggtitle("4. rose dataset"))

#Selecting the right approach
TrainingData <- upsampled_dataset


############################################
#Model Training
############################################

#Create Common Train contol 
trainControl_common <- trainControl(method = "repeatedcv", number = 30, repeats = 5)
set.seed(100)

# MODEL 1. Support vector machine
#--------------------------------------
set.seed(100)
trainControl_mod1 <- trainControl_common

#Remove the 'EmployeeNumber' to refrain having impact on model
Model1_SVM <- train(Attrition ~., data = TrainingData[,-8], method = "svmLinear",
                    trControl=trainControl_mod1,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
Model1_SVM
summary(Model1_SVM)

TestingData_Mod1 <- TestingData[,!(names(TestingData) %in% c("Attrition"))]
test_pred_mod1 <- predict(Model1_SVM, newdata = TestingData_Mod1)

Model1_SVM_CM <- confusionMatrix(test_pred_mod1,TestingData$Attrition,
                                 positive = 'Yes', mode = 'everything')
Model1_SVM_CM


Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(test_pred_mod1=="Yes",1,0)

# AUC
Model1_AUC <- auc(Validate, predicted)
Model1_AUC

#ROC
Model1_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(test_pred_mod1),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model1_ROC


# MODEL 2. Random Forest
#--------------------------------------
set.seed(100)
trainControl_mod2 <- trainControl_common

#Remove the 'EmployeeNumber' to refrain having impact on model 
#Model2_RF <- train(Attrition ~., data = TrainingData[,-8], method = "rf",
#                    trControl=trainControl_mod2,
#                    preProcess = c("center", "scale"),
#                    tuneLength = 10)

Model2_RF <- randomForest(Attrition ~ ., data=TrainingData[,-8], ntree=2000,  mtry=4,
                          importance=TRUE, replace=FALSE)

Model2_RF
plot(Model2_RF, main="")
legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest")

TestingData_Mod2 <- TestingData[,!(names(TestingData) %in% c("Attrition"))] 
pred_Test_mod2 <- predict(Model2_RF, newdata = TestingData_Mod2)

result_Test <- confusionMatrix(TestingData$Attrition, pred_Test_mod2)
result_Test

Model2_RF_CM <- confusionMatrix(data = pred_Test_mod2,
                                reference = TestingData$Attrition,
                                positive = 'Yes', mode = 'everything')
Model2_RF_CM

Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(pred_Test_mod2=="Yes",1,0)

# AUC
Model2_AUC <- auc(Validate, predicted)
Model2_AUC

#ROC
Model2_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(pred_Test_mod2),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model2_ROC

# MODEL 3. Exterme Gradient Boost
#--------------------------------------

set.seed(100)
trainControl_mod3 <- trainControl_common
trainControl_mod3$classProbs = TRUE

xgbGrid <- expand.grid(nrounds = 50,
                       max_depth = 12,
                       eta = .03,
                       gamma = 0.01,
                       colsample_bytree = .7,
                       min_child_weight = 1,
                       subsample = 0.9)

Model3_XGE <- train(Attrition ~ ., data = TrainingData[,-8],
                    method = "xgbTree"
                    ,trControl = trainControl_mod3
                    , verbose=0
                    , maximize=FALSE
                    ,tuneGrid = xgbGrid)

Model3_XGE

pred_Test_mod3 <- predict(object = Model3_XGE, TestingData,
                          type = 'raw')
Model3_XGE_CM <- confusionMatrix(data = pred_Test_mod3,
                                 reference = TestingData$Attrition,
                                 positive = 'Yes', mode = 'everything')
Model3_XGE_CM

Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(pred_Test_mod3=="Yes",1,0)

# AUC
Model3_AUC <- auc(Validate, predicted)
Model3_AUC

#ROC
Model3_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(pred_Test_mod3),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model3_ROC


#ROC
Model3_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(test_pred_mod3),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model3_ROC

MODEL 4. Gradient Boost
--------------------------------------
set.seed(100)
#trainControl_mod4 <- trainControl(method = 'cv', number = 3,
#                           returnResamp='none',
#                           summaryFunction = twoClassSummary,
#                           classProbs = TRUE)
trainControl_mod4 <- trainControl_common
trainControl_mod4$classProbs = TRUE

#Remove the 'EmployeeNumber' to refrain having impact on model
Model4_GT <- train(Attrition ~ ., data = TrainingData[,-8],
                   method = 'gbm',
                   trControl = trainControl_mod4,
                   metric = "ROC",
                   preProc = c("center", "scale"))
summary(Model4_GT)

pred_Test_mod4 <- predict(object = Model4_GT, TestingData,
                          type = 'raw')
Model4_GT_CM <- confusionMatrix(data = pred_Test_mod4,
                                reference = TestingData$Attrition,
                                positive = 'Yes', mode = 'everything')
Model4_GT_CM

Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(pred_Test_mod4=="Yes",1,0)

# AUC
Model4_AUC <- auc(Validate, predicted)
Model4_AUC

#ROC
Model4_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(pred_Test_mod4),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model4_ROC


MODEL 5. Neural Network
--------------------------------------
set.seed(100)
trainControl_mod5 <- trainControl_common

#Remove the 'EmployeeNumber' to refrain having impact on model
Model5_NN <- nnet(Attrition~., data = TrainingData[,-8],size = 5,maxit = 2000,decay = .01)
options(scipen = 99)

Model5_NN

head(Model5_NN$fitted.values)
plotnet(Model5_NN)

test_pred_mod5 <- predict(Model5_NN,TestingData,type = "raw")
#h refers to limit - employees have probability than 0.5 will be selected as limit
cutoff <- floor(test_pred_mod5 +.5 )
plot(test_pred_mod5)
abline(a=0.5,b=0,h=0.5)

nn_result = ifelse(cutoff==1,"Yes","No")
nn_result <- as.factor(nn_result)
Model5_NN_CM <- confusionMatrix(nn_result, reference = TestingData$Attrition,positive = "Yes", mode = "everything")
Model5_NN_CM

pred_NN <- ROCR::prediction(cutoff,TestingData$Attrition)
performance_NN <- performance(pred_NN,"tpr","fpr")
plot(performance_NN, main = "Performance - Neural Network")

Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(nn_result=="Yes",1,0)

# AUC
Model5_AUC <- auc(Validate, predicted)
Model5_AUC

#ROC
Model5_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(test_pred_mod5),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model5_ROC


# MODEL 6. Adaboost
#--------------------------------------
set.seed(100)

trainControl_mod6 <- rpart::rpart.control(maxdepth=30,
                                          cp=0.010000,
                                          minsplit=20,
                                          xval=10)

#Remove the 'EmployeeNumber' to refrain having impact on model
Model6_ADA <- ada(Attrition ~ .,
                  data=TrainingData[,-8],
                  control=trainControl_mod6, iter=50)

Model6_ADA

test_pred_mod6<-predict(Model6_ADA,TestingData)


Model6_ADA_CM <- confusionMatrix(test_pred_mod6, reference = TestingData$Attrition,positive = "Yes", mode = "everything")
Model6_ADA_CM

Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(test_pred_mod6=="Yes",1,0)

# AUC
Model6_AUC <- auc(Validate, predicted)
Model6_AUC

#ROC
Model6_ROC <- plot.roc (as.numeric(TestingData$Attrition), as.numeric(test_pred_mod6),lwd=2, type="b", print.auc=TRUE, col ="blue")
Model6_ROC

# MODEL 7. Decision Tree
#--------------------------------------
set.seed(100)
Model7_DT <- rpart(Attrition ~ ., data=TrainingData, method="class", parms=list(split="information"),
                   control=rpart.control(usesurrogate=1,  maxsurrogate=1))

#Remove the 'EmployeeNumber' to refrain having impact on model
#Model7_DT  <- tree::tree (Attrition ~., data = TrainingData[,-8])
plot(Model7_DT)
text(Model7_DT , all = T)

Model7_DT
summary(Model3_DT)

test_pred_mod7<-predict(Model7_DT,TestingData, type="class")

Model7_DT_CM <- confusionMatrix(data = test_pred_mod7, #test_pred_mod3[,1]
                                reference = TestingData$Attrition,
                                mode = 'everything')
Model7_DT_CM


Validate<-ifelse(TestingData$Attrition=="Yes",1,0)
predicted<-ifelse(test_pred_mod7=="Yes",1,0)

# AUC
Model7_AUC <- auc(Validate, predicted)
Model7_AUC


# MODEL 8. Ensemble Models
#--------------------------------------

set.seed(100)
trainControl_mod8 <- trainControl(method="repeatedcv",
                                  number=3,
                                  repeats=1,
                                  search="random",
                                  summaryFunction=twoClassSummary,
                                  classProbs=TRUE,
                                  savePredictions=TRUE)

#Remove the 'EmployeeNumber' to refrain having impact on model
model_list_ensemble <- caretList(Attrition ~ .,
                                 data=TrainingData[,-8],
                                 trControl=trainControl_mod8,
                                 methodList=c("svmRadial", "rf", "xgbLinear"))

Model_Ensemble <- caretStack(
  model_list_ensemble,
  metric="ROC",
  method="glm",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)


############################################
#Evaluation & Hybrid model
############################################

#models_comparion <- resamples(list(SVM = Model1_SVM, RF = Model2_RF, XGE = Model3_XGE, GT = Model4_GT, NN = Model5_NN, ADA = Model6_ADA, DT = Model7_DT))
models_comparion <- resamples(list(SVM = Model1_SVM, RF = Model2_RF, XGE = Model3_XGE, GT = Model4_GT))
dotplot(models_comparion)

models_to_evaluate <- list(Model1_SVM, Model2_RF, Model3_XGE, Model4_GT, Model6_ADA)
#models_to_evaluate <- list(Model1_SVM, Model2_RF, Model3_XGE, Model4_GT, Model5_NN, Model6_ADA, Model7_DT)

predictions <-lapply(models_to_evaluate,  predict,
                     newdata=select(TestingData, -Attrition))

Model_Ensemble_CM <- lapply(predictions,
                            confusionMatrix,
                            reference=TestingData$Attrition,
                            positive="Yes")
Model_Ensemble_CM

# accuracy
Model_Ensemble_accuracy_metrics <-
  lapply(Model_Ensemble_CM, `[[`, "overall") %>%
  lapply(`[`, 1) %>%
  unlist()

# recall
Model_Ensemble_rec_metrics <-
  lapply(Model_Ensemble_CM, `[[`, "byClass") %>%
  lapply(`[`, 1) %>%
  unlist()

# precision
Model_Ensemble_pre_metrics <-
  lapply(Model_Ensemble_CM, `[[`, "byClass") %>%
  lapply(`[`, 3) %>%
  unlist()


Model_Ensemble_accuracy_metrics
Model_Ensemble_rec_metrics
Model_Ensemble_pre_metrics

performance_time_Model1_SVM <- system.time(Model1_SVM)
performance_time_Model2_RF <- system.time(Model2_RF)
performance_time_Model3_XGE <- system.time(Model3_XGE)
performance_time_Model4_GT <- system.time(Model4_GT)
performance_time_Model5_NN <- system.time(Model5_NN)
performance_time_Model6_ADA <- system.time(Model6_ADA)
performance_time_Model7_DT <- system.time(Model7_DT)


algorithm_model_list <- c("Model 1 -  SVM", "Model 2 - RF", "Model 3 - XGE", "Model 4 - GT", "Model 6 - ADA")

model_comparions_result <-
  data.frame(Models=algorithm_model_list,
             Accuracy=Model_Ensemble_accuracy_metrics,
             Recall=Model_Ensemble_rec_metrics,
             Precision=Model_Ensemble_pre_metrics)

model_comparions_result

#----------------------------------------------
models_to_evaluate <- list(Model2_RF, Model3_XGE, Model6_ADA)
#models_to_evaluate <- list(Model1_SVM, Model2_RF, Model3_XGE, Model4_GT, Model5_NN, Model6_ADA, Model7_DT)

predictions <-lapply(models_to_evaluate,  predict, 
                     newdata=select(TestingData, -Attrition))

Model_Ensemble_CM <- lapply(predictions,
                            confusionMatrix, 
                            reference=TestingData$Attrition, 
                            positive="Yes")
Model_Ensemble_CM


############################################
#Re-evaluation & result
############################################

# Creating Hybrid model (Weighted Average) 

#-----------------------------------------
ensemble_model_data = cbind(pred_Test_mod2, pred_Test_mod3, target=TestingData$Attrition)
names(ensemble_model_data) = c("RF", "XGB", "target")

ensemble_model_data <- as.data.frame(ensemble_model_data) #model_ensembling[-4]
descrCor <- cor(ensemble_model_data[-3])
descrCor

#Applying Logistic Regression

ensemble_model_data[ensemble_model_data == 1] <- 0 
ensemble_model_data[ensemble_model_data == 2] <- 1
ensemble_model_data

mylogistic <- glm(target ~ ., data = ensemble_model_data, family = "binomial")
summary(mylogistic)$coefficient
log_sum = data.frame(summary(mylogistic)$coefficient)

#Clean the coefficients
log_sum$variables = row.names(log_sum)
log_sum= log_sum[c("variables", "Estimate")][-1,]
log_sum

#Calculate and applying Weights
typeof(log_sum$weight)
log_sum$weight <- as.numeric(xx$weight)
log_sum$weight=abs(log_sum$weight)

log_sum$weight= log_sum$Estimate / sum(log_sum$Estimate)

ensemble_model_data$EnsemblePred = (log_sum$weight[1]*ensemble_model_data$pred_Test_mod2) + (log_sum$weight[2]*ensemble_model_data$pred_Test_mod3) 

#Measuring performance
perf = prediction(ensemble_model_data$EnsemblePred, ensemble_model_data$target)
auc = performance(perf, "auc")
auc

ensemble_model_data$EnsemblePred <- as.integer(ensemble_model_data$EnsemblePred)
TestingData$Attrition <- ifelse(TestingData$Attrition == "Yes",1,0)
FinalHRModel_CM <- confusionMatrix(factor(ensemble_model_data$EnsemblePred), factor(TestingData$Attrition))
FinalHRModel_CM

FinalHRModel_CM_Accuracy <- FinalHRModel_CM$overall['Accuracy']
FinalHRModel_CM_Kappa <- FinalHRModel_CM$overall['Kappa']
FinalHRModel_CM_Sensitivity <- FinalHRModel_CM$byClass['Sensitivity']
FinalHRModel_CM_Specificity <- FinalHRModel_CM$byClass['Specificity']
FinalHRModel_CM_Precision <- FinalHRModel_CM$byClass['Precision']
FinalHRModel_CM_Recall<- FinalHRModel_CM$byClass['Recall']
FinalHRModel_CM_F1 <- FinalHRModel_CM$byClass['F1']

# AUC
FinalHRModel_CM_AUC <- auc(ensemble_model_data$EnsemblePred, ensemble_model_data$target)
FinalHRModel_CM_AUC

FinalHRModel_CM_ROC_PLOT <- plot.roc (as.numeric(ensemble_model_data$target), as.numeric(ensemble_model_data$EnsemblePred),lwd=2, type="b", print.auc=TRUE, col ="blue")
FinalHRModel_CM_ROC_PLOT


FinalHRModel_CM_TABLE <- data.frame(
  Accuracy=rep(c(FinalHRModel_CM_Accuracy)),
  Kappa=rep(c(FinalHRModel_CM_Kappa)),
  Sensitivity=rep(c(FinalHRModel_CM_Sensitivity)),
  Specificity=rep(c(FinalHRModel_CM_Specificity)),
  Precision=rep(c(FinalHRModel_CM_Precision)),
  Recall=rep(c(FinalHRModel_CM_Recall)),
  F1=rep(c(FinalHRModel_CM_F1)),
  AUC = rep(c(FinalHRModel_CM_AUC)))


ListOfEmployeesLeaving <- data.frame(TestingData[ensemble_model_data$EnsemblePred==1,c(8,32)])
ListOfEmployeesLeaving

#write to file
Result <-setNames(ListOfEmployeesLeaving,c("Employees with leaving tendancy"))
write.csv(Result,"HR-Result.csv")

############################################
#UI Creation
############################################

ui <- fluidPage(
  
  theme = shinytheme("cosmo"), #bslib::bs_theme(bootswatch = "darkly"), #shinytheme("cosmo"),superhero
  titlePanel("Welcome to employee predictor ..."),
  h3("Be saviour to your workforce"),
  sidebarLayout(
    sidebarPanel(
      
      h6("Thanks for choosing HR predictor. "),
      #shiny::img(src = "cover.png",height = 200,width = 400),
      h3("Creator : "),
      h2("Jayasai Goutheman")
      
    ),
    mainPanel(
      tabsetPanel(
        #Tab 1
        tabPanel("Data Exploration", h3("Here is your raw data Before Imbalance Correction:"),DT::dataTableOutput("rawData")),
        
        tabPanel("Descriptive Analysis",
                 tabsetPanel(tabPanel("Attrition Analysis",
                                      h3("Attrition data based on your data (Before Imbalance Correction)"),
                                      fluidRow(column(5,plotOutput("attr_bar")),column(5,plotOutput("attr_pie")))
                 ),
                 
                 
                 tabPanel("Dataset Attributes",
                          
                          selectInput(
                            "var_quant_univ", "",
                            c('Age','DistanceFromHome','MonthlyIncome','NumCompaniesWorked',
                              'PercentSalaryHike','TotalWorkingYears',
                              'YearsAtCompany','YearsSinceLastPromotion',
                              'YearsWithCurrManager')
                          ),
                          
                          h3("Attributes in your HR data, for your glance ..."),
                          
                          fluidRow(column(5,plotOutput("univ_hist"))),
                          
                          selectInput(
                            "var_qual_univ", "",
                            c("JobInvolvement", "PerformanceRating", "EnvironmentSatisfaction",
                              "JobSatisfaction" ,"WorkLifeBalance", "BusinessTravel",
                              "Department", "Education", "EducationField", "Gender", "JobLevel",
                              "JobRole","MaritalStatus")
                          ),
                          
                          fluidRow(column(5,plotOutput("univ_bar")),column(5,plotOutput("univ_pie"))),
                          
                 ),
                 
                 tabPanel("Attribute Comparision (Advanced)",
                          h3("Compare attributes with each other ..."),
                          fluidRow(column(10,plotOutput("biv_corrplot"))),
                 )
                 )),
        
        #Tab 3
        tabPanel("Imbalance Correction (technical)",h5("Imbalance correction allows to get better result in imbalance dataset."), h1("Meathod - Up scaling"),
                 tabsetPanel(tabPanel("Before",
                                      h3("Attrition data based on your data (Before Imbalance Correction)"),
                                      textOutput("before_imbalace_count"),
                                      h3(""),
                                      fluidRow(column(5,plotOutput("attr_bar_1")),column(5,plotOutput("attr_pie_1")))
                 ),
                 
                 tabPanel("After",
                          h3("Attrition data based on your data (After Imbalance Correction)"),
                          textOutput("after_imbalace_count"),
                          h3(""),
                          fluidRow(column(5,plotOutput("attr_bar_after")),column(5,plotOutput("attr_pie_after")))
                 )
                 
                 
                 )),
        
        #Tab 4
        tabPanel("Predication",h3("Following are the employees might leave"), 
                 tabsetPanel(tabPanel("Result", 
                                      h3(""),
                                      DT::dataTableOutput("displayResult")
                 ),
                 
                 tabPanel("Evaluation Matrix (Technical)",
                          h3(""),
                          DT::dataTableOutput("displayEvalationMetrics"),
                          plotOutput("final_model_roc")
                 )
                 
                 
                 ))
        
      )
      
    )
  )
  
)


########## Server Building for Shiny App #################

server <- function(input, output) {
  
  output$before_imbalace_count <- renderText({ 
    paste("Total number of  records", nrow(dataset), "(Training + Testing)", sep=" ")
  }) 
  
  output$after_imbalace_count <- renderText({ 
    paste("Total number of  records", nrow(upsampled_dataset), "(Training)", sep=" ")
  })
  
  output$rawData <-  DT::renderDataTable({
    dataset
  })
  
  output$trainingData <-  DT::renderDataTable({
    TrainingData
  })
  
  output$afterImbalancedData <-  DT::renderDataTable({
    upsampled_dataset
  })
  
  output$displayResult <-  DT::renderDataTable({
    ListOfEmployeesLeaving #Hide the row number
  })
  
  output$displayEvalationMetrics <-  DT::renderDataTable({
    FinalHRModel_CM_TABLE
  })
  
  output$univ_bar <- renderPlot({
    barPlotUniv(get_quali(), input$var_qual_univ, FALSE)
  })
  
  output$univ_pie <- renderPlot({
    piePlotUniv(get_quali(), input$var_qual_univ)
  })
  
  output$attr_bar <- renderPlot({
    barPlotUniv(data.frame(dataset), "Attrition")
  })
  
  output$attr_bar_1 <- renderPlot({
    barPlotUniv(data.frame(dataset), "Attrition")
  })
  
  output$attr_bar_after <- renderPlot({
    barPlotUniv(data.frame(upsampled_dataset), "Attrition")
  })
  
  output$attr_pie <- renderPlot({
    piePlotUniv(dataset, "Attrition")
  })
  
  output$attr_pie_1 <- renderPlot({
    piePlotUniv(dataset, "Attrition")
  })
  
  output$attr_pie_after <- renderPlot({
    piePlotUniv(upsampled_dataset, "Attrition")
  })
  
  output$final_model_roc <- renderPlot({
    FinalHRModel_CM_ROC_PLOT
  })
  
  output$biv_corrplot <- renderPlot({
    cor_m = cor(get_quanti()[,1:11])
    corrplot(cor_m, method="color", title = "", order="hclust")
  })
  
  #Chart functions
  #----------------------------------
  get_quali <- reactive({
    is.fact <- sapply(dataset, is.factor)
    quali = dataset[,is.fact]
    as.data.frame(quali)
  })
  
  get_quanti <- reactive({
    is.fact <- sapply(dataset, is.factor)
    quanti = dataset[, !is.fact]
    quanti$Attrition = dataset$Attrition
    as.data.frame(quanti)
    
  })
  
  output$univ_hist <- renderPlot({
    histoUniv(get_quanti(), input$var_quant_univ, FALSE)
  })
  
  histoUniv = function(df, var, density=FALSE){
    if (density){
      p = ggplot(data = df, aes(x=df[,var], y=..density..)) +
        geom_histogram(position="identity", fill=4) +
        geom_density(alpha=.2) +
        labs(x=var, y="Densité")
    }else{
      p = ggplot(data = df, aes(x=df[,var])) +
        geom_histogram(position="identity", fill=4) +
        labs(x=var, y="Count")
    }
    
    p+geom_vline(xintercept = mean(df[,var]), linetype="dashed")+
      ggtitle(paste("Histogram Analysis : ", var, sep=""))
  }
  
  piePlotUniv = function(df, var){
    cc = table(df[,var])
    pie(cc, main = paste("Pie-Chart : ", var, sep=""))
  }
  
  barPlotUniv = function(df, var, pourcent=FALSE){
    if (pourcent){
      res = ggplot(data=df, aes(x=df[,var], y=..count../sum(..count..), fill=4)) +
        labs(y="Percentage : ", x = var)
      
    }else{
      res = ggplot(data=df, aes(x=df[,var], fill=4)) +
        labs(y="Count", x = var)
    }
    
    
    res + geom_bar(stat="count") +
      ggtitle(paste("Barplot : ", var, sep="")) + 
      guides(fill = FALSE) 
    
  }
  
}


#### Calling UI Server for Rendering App ###########

shinyApp(ui = ui, server = server) 
