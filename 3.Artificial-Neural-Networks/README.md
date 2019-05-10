---
title: "Educational data Mining using ANN"
author: "Bharath Karumudi"
date: "5/6/2019"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



### Introduction
A kaggle dataset on student academic performance (available through https://www.kaggle.com/aljarah/xAPI-Edu-Data) is gathered to identify the influential factors for students’ performance.

To predict the students’ performance, the collected data was organized into four kinds of features: demographic, academic background, parents’ participation on learning process and behavioral features.

### Objective
Experiment with different ANN architectural parameters (e.g. number of hidden layers, number of nodes within each layer) as well as model parameters (activation and loss functions, regularization, epoch/batch size, etc.). Evaluate and report the performance of ANN models.

### Install required Packages
    (if needed)


```r
# install.packages("ggplot2")
# install.packages("lattice")
# install.packages("caret")
# install.packages("C50")
# install.packages("randomForest")
# install.packages("nnet")
# install.packages("RRF")
```

### Load the required libraries


```r
library(ggplot2)
library(lattice)
library(caret)
library(C50)
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(nnet)
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## 
## Attaching package: 'rattle'
```

```
## The following object is masked from 'package:randomForest':
## 
##     importance
```

```r
library(RRF)
```

```
## RRF 1.9
```

```
## Type rrfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'RRF'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following objects are masked from 'package:randomForest':
## 
##     classCenter, combine, getTree, grow, importance, margin,
##     MDSplot, na.roughfix, outlier, partialPlot, treesize,
##     varImpPlot, varUsed
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

### Loading the dataset


```r
educational_dataset <- read.csv("Students_Academic_Performance.csv" )

cat("Let's see the data:")
```

```
## Let's see the data:
```

```r
dim(educational_dataset)
```

```
## [1] 480  17
```

```r
head(educational_dataset)
```

```
##   gender NationalITy PlaceofBirth    StageID GradeID SectionID Topic
## 1      M          KW       KuwaIT lowerlevel    G-04         A    IT
## 2      M          KW       KuwaIT lowerlevel    G-04         A    IT
## 3      M          KW       KuwaIT lowerlevel    G-04         A    IT
## 4      M          KW       KuwaIT lowerlevel    G-04         A    IT
## 5      M          KW       KuwaIT lowerlevel    G-04         A    IT
## 6      F          KW       KuwaIT lowerlevel    G-04         A    IT
##   Semester Relation raisedhands VisITedResources AnnouncementsView
## 1        F   Father          15               16                 2
## 2        F   Father          20               20                 3
## 3        F   Father          10                7                 0
## 4        F   Father          30               25                 5
## 5        F   Father          40               50                12
## 6        F   Father          42               30                13
##   Discussion ParentAnsweringSurvey ParentschoolSatisfaction
## 1         20                   Yes                     Good
## 2         25                   Yes                     Good
## 3         30                    No                      Bad
## 4         35                    No                      Bad
## 5         50                    No                      Bad
## 6         70                   Yes                      Bad
##   StudentAbsenceDays Class
## 1            Under-7     M
## 2            Under-7     M
## 3            Above-7     L
## 4            Above-7     L
## 5            Above-7     M
## 6            Above-7     M
```

```r
summary(educational_dataset)
```

```
##  gender     NationalITy       PlaceofBirth         StageID   
##  F:175   KW       :179   KuwaIT     :180   HighSchool  : 33  
##  M:305   Jordan   :172   Jordan     :176   lowerlevel  :199  
##          Palestine: 28   Iraq       : 22   MiddleSchool:248  
##          Iraq     : 22   lebanon    : 19                     
##          lebanon  : 17   SaudiArabia: 16                     
##          Tunis    : 12   USA        : 16                     
##          (Other)  : 50   (Other)    : 51                     
##     GradeID    SectionID     Topic     Semester   Relation  
##  G-02   :147   A:283     IT     : 95   F:245    Father:283  
##  G-08   :116   B:167     French : 65   S:235    Mum   :197  
##  G-07   :101   C: 30     Arabic : 59                        
##  G-04   : 48             Science: 51                        
##  G-06   : 32             English: 45                        
##  G-11   : 13             Biology: 30                        
##  (Other): 23             (Other):135                        
##   raisedhands     VisITedResources AnnouncementsView   Discussion   
##  Min.   :  0.00   Min.   : 0.0     Min.   : 0.00     Min.   : 1.00  
##  1st Qu.: 15.75   1st Qu.:20.0     1st Qu.:14.00     1st Qu.:20.00  
##  Median : 50.00   Median :65.0     Median :33.00     Median :39.00  
##  Mean   : 46.77   Mean   :54.8     Mean   :37.92     Mean   :43.28  
##  3rd Qu.: 75.00   3rd Qu.:84.0     3rd Qu.:58.00     3rd Qu.:70.00  
##  Max.   :100.00   Max.   :99.0     Max.   :98.00     Max.   :99.00  
##                                                                     
##  ParentAnsweringSurvey ParentschoolSatisfaction StudentAbsenceDays Class  
##  No :210               Bad :188                 Above-7:191        H:142  
##  Yes:270               Good:292                 Under-7:289        L:127  
##                                                                    M:211  
##                                                                           
##                                                                           
##                                                                           
## 
```

### RFE Model
We can identify the attributes that are not required by using Recursive Feature Elimination method (RFE).


```r
set.seed(1234)

rfe_control_params <- rfeControl(functions=rfFuncs, method="cv", number=10)

rfe_method<- rfe(educational_dataset[,1:16], educational_dataset[,17], sizes=c(1:17), rfeControl=rfe_control_params)
print(rfe_method)
```

```
## 
## Recursive feature selection
## 
## Outer resampling method: Cross-Validated (10 fold) 
## 
## Resampling performance over subset size:
## 
##  Variables Accuracy  Kappa AccuracySD KappaSD Selected
##          1   0.5208 0.2614    0.02631 0.03523         
##          2   0.6457 0.4630    0.04099 0.05403         
##          3   0.6896 0.5257    0.05226 0.07535         
##          4   0.7141 0.5620    0.06140 0.09099         
##          5   0.7100 0.5539    0.04795 0.07220         
##          6   0.7395 0.5987    0.06335 0.09745         
##          7   0.7559 0.6233    0.06157 0.09386         
##          8   0.7582 0.6253    0.04144 0.06600         
##          9   0.7604 0.6284    0.05051 0.07584         
##         10   0.7956 0.6835    0.04889 0.07472         
##         11   0.7872 0.6706    0.04838 0.07437         
##         12   0.7956 0.6830    0.05748 0.08848         
##         13   0.7915 0.6757    0.04281 0.06598         
##         14   0.7851 0.6662    0.03986 0.06206         
##         15   0.8082 0.7024    0.04792 0.07475        *
##         16   0.7976 0.6858    0.04990 0.07692         
## 
## The top 5 variables (out of 15):
##    StudentAbsenceDays, VisITedResources, raisedhands, AnnouncementsView, ParentAnsweringSurvey
```

```r
predictors(rfe_method)
```

```
##  [1] "StudentAbsenceDays"       "VisITedResources"        
##  [3] "raisedhands"              "AnnouncementsView"       
##  [5] "ParentAnsweringSurvey"    "Relation"                
##  [7] "Discussion"               "Topic"                   
##  [9] "NationalITy"              "ParentschoolSatisfaction"
## [11] "gender"                   "PlaceofBirth"            
## [13] "GradeID"                  "StageID"                 
## [15] "SectionID"
```

```r
plot(rfe_method, type=c("g", "o"))
```

![](Q2_files/figure-html/RFE_model-1.png)<!-- -->

### Train an rpart model to compute variable importance

```r
# Train an rpart model and compute variable importance.
set.seed(100)
rPartMod <- train(Class ~ ., data=educational_dataset, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)
```

```
## rpart variable importance
## 
##   only 20 most important variables shown (out of 61)
## 
##                              Overall
## VisITedResources              100.00
## raisedhands                    83.50
## AnnouncementsView              74.62
## StudentAbsenceDaysUnder-7      56.96
## ParentAnsweringSurveyYes       43.45
## RelationMum                    16.92
## ParentschoolSatisfactionGood   16.12
## NationalITyIran                 0.00
## PlaceofBirthUSA                 0.00
## StageIDlowerlevel               0.00
## NationalITyJordan               0.00
## NationalITyIraq                 0.00
## `GradeIDG-12`                   0.00
## PlaceofBirthKuwaIT              0.00
## TopicGeology                    0.00
## NationalITylebanon              0.00
## PlaceofBirthMorocco             0.00
## `GradeIDG-10`                   0.00
## TopicIT                         0.00
## NationalITyMorocco              0.00
```


### Train an RRf model to compute variable importance 

```r
set.seed(1234)
rrfModel <- train(Class ~ ., data=educational_dataset, method="RRF")
rrfImp <- varImp(rrfModel, scale=F)
rrfImp
```

```
## RRF variable importance
## 
##   only 20 most important variables shown (out of 60)
## 
##                              Overall
## VisITedResources              77.417
## StudentAbsenceDaysUnder-7     53.795
## raisedhands                   37.892
## AnnouncementsView             30.954
## Discussion                    21.755
## RelationMum                   13.461
## ParentAnsweringSurveyYes       8.063
## genderM                        6.833
## ParentschoolSatisfactionGood   4.223
## NationalITySaudiArabia         3.981
## NationalITyJordan              2.608
## TopicChemistry                 2.492
## TopicMath                      2.426
## TopicGeology                   2.352
## StageIDlowerlevel              2.228
## TopicEnglish                   2.187
## SectionIDB                     2.117
## GradeIDG-06                    2.097
## GradeIDG-08                    2.062
## PlaceofBirthJordan             1.948
```

```r
plot(rrfImp, top = 20, main='Variable Importance')
```

![](Q2_files/figure-html/rrf_model-1.png)<!-- -->

### Pre-processing the data


```r
# Check if there are any na values in the dataset.
sum(is.na(educational_dataset))
```

```
## [1] 0
```

```r
correlationMatrix <- cor(educational_dataset[,10:13])
correlationMatrix
```

```
##                   raisedhands VisITedResources AnnouncementsView
## raisedhands         1.0000000        0.6915717         0.6439178
## VisITedResources    0.6915717        1.0000000         0.5945000
## AnnouncementsView   0.6439178        0.5945000         1.0000000
## Discussion          0.3393860        0.2432918         0.4172900
##                   Discussion
## raisedhands        0.3393860
## VisITedResources   0.2432918
## AnnouncementsView  0.4172900
## Discussion         1.0000000
```

```r
findCorrelation(correlationMatrix, cutoff=0.75)
```

```
## integer(0)
```

```r
# Removing the semester attribute
educational_data <- educational_dataset[,-8]
```


### Sampling the data into Train and Test Datasets
    using stratified sampling of 70% - 30%.


```r
set.seed(1234)
#Stratified Sampling 70%
TrainingDataIndex <- createDataPartition(educational_data$Class, p=0.70, list = FALSE)

#Training Data
training_data <- educational_data[TrainingDataIndex,]

#Test Data
test_data <- educational_data[-TrainingDataIndex,]
```



### Building the Model with Artifical Neural Networks


```r
#Train Params
trcontrolparams <- trainControl(method = "repeatedcv", number = 5, repeats=8)

## Building the Model with Neural Networks

ANN_model <- train(training_data[,-17], training_data$Class,
                   method = "nnet",
                   trControl = trcontrolparams,
                   preProcess = c("scale", "center"),
                   na.action =  na.omit)

modelLookup('nnet')
```


### Test the model with test dataset

```r
ANN_predict <- predict(ANN_model, test_data)
```


### Validate the results

```r
confusionMatrix(ANN_predict, test_data$Class)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  H  L  M
##          H 42  0  0
##          L  0 38  0
##          M  0  0 63
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9745, 1)
##     No Information Rate : 0.4406     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: H Class: L Class: M
## Sensitivity            1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000
## Prevalence             0.2937   0.2657   0.4406
## Detection Rate         0.2937   0.2657   0.4406
## Detection Prevalence   0.2937   0.2657   0.4406
## Balanced Accuracy      1.0000   1.0000   1.0000
```


### Conclusion

With the selected parameters, the ANN model is working with an accuracy of 1. The corelation matrix also given us the association between the variable.With this model it will help in predicting how students will perform based on the attributes and also what impacts them to fail.


### References
1. RRF documentation
2. Randomforest documentation
3. Artificial Neural Networks (ANN) library documentation
4. Coorelation Matrix
