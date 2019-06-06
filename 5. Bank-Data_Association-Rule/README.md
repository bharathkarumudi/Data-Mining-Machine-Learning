---
title: "Bank Data ~ Association Rules Mining"
author: "Bharath Karumudi"
date: "6/2/2019"
output:
  html_document:
    keep_md: true
---



### Introduction:  
A dataset of a bank was given and need to explore the bank data (bankdata.csv) and an accompanying description of the attributes and their values are available in (bankdataDescription.doc). The dataset contains attributes on each person’s demographics and banking information in order to determine they will want to obtain the new PEP (Personal Equity Plan).

### Objective:  
Objective is to perform Association Rule discovery on the dataset and describe the association rule mining process and the resulting 5 interesting rules, each with their three items of explanation and recommendations. For at least one of the rules, discuss the support, confidence and lift values and how they are interpreted in this data set.

### Install Required Packages:  

```r
#install.packages("arules")
#install.packages("arulesViz")
```

### Load Libraries: 

```r
library(arules)
```

```
## Loading required package: Matrix
```

```
## 
## Attaching package: 'arules'
```

```
## The following objects are masked from 'package:base':
## 
##     abbreviate, write
```

```r
library(arulesViz)
```

```
## Loading required package: grid
```

### Load Data: 

```r
data <- read.csv('Files/bankdata.csv')
str(data)
```

```
## 'data.frame':	600 obs. of  12 variables:
##  $ id         : Factor w/ 600 levels "ID12101","ID12102",..: 1 2 3 4 5 6 7 8 9 10 ...
##  $ age        : int  48 40 51 23 57 57 22 58 37 54 ...
##  $ sex        : Factor w/ 2 levels "FEMALE","MALE": 1 2 1 1 1 1 2 2 1 2 ...
##  $ region     : Factor w/ 4 levels "INNER_CITY","RURAL",..: 1 4 1 4 2 4 2 4 3 4 ...
##  $ income     : num  17546 30085 16575 20375 50576 ...
##  $ married    : Factor w/ 2 levels "NO","YES": 1 2 2 2 2 2 1 2 2 2 ...
##  $ children   : int  1 3 0 3 0 2 0 0 2 2 ...
##  $ car        : Factor w/ 2 levels "NO","YES": 1 2 2 1 1 1 1 2 2 2 ...
##  $ save_act   : Factor w/ 2 levels "NO","YES": 1 1 2 1 2 2 1 2 1 2 ...
##  $ current_act: Factor w/ 2 levels "NO","YES": 1 2 2 2 1 2 2 2 1 2 ...
##  $ mortgage   : Factor w/ 2 levels "NO","YES": 1 2 1 1 1 1 1 1 1 1 ...
##  $ pep        : Factor w/ 2 levels "NO","YES": 2 1 1 1 1 2 2 1 1 1 ...
```

### Data Cleansing: 

```r
#Converting age and income to Categorical Variables
data$income.bracket <- cut(data$income, 3, labels = c("low", "med", "high"))
data$age.group <- cut(data$age, 3) 

#The ID column is not required
data <- subset(data, select=-c(id))

data$children <- as.factor(data$children)
data <- data[,sapply(data, is.factor)]

#Lets see the cleansed dataset.
str(data)
```

```
## 'data.frame':	600 obs. of  11 variables:
##  $ sex           : Factor w/ 2 levels "FEMALE","MALE": 1 2 1 1 1 1 2 2 1 2 ...
##  $ region        : Factor w/ 4 levels "INNER_CITY","RURAL",..: 1 4 1 4 2 4 2 4 3 4 ...
##  $ married       : Factor w/ 2 levels "NO","YES": 1 2 2 2 2 2 1 2 2 2 ...
##  $ children      : Factor w/ 4 levels "0","1","2","3": 2 4 1 4 1 3 1 1 3 3 ...
##  $ car           : Factor w/ 2 levels "NO","YES": 1 2 2 1 1 1 1 2 2 2 ...
##  $ save_act      : Factor w/ 2 levels "NO","YES": 1 1 2 1 2 2 1 2 1 2 ...
##  $ current_act   : Factor w/ 2 levels "NO","YES": 1 2 2 2 1 2 2 2 1 2 ...
##  $ mortgage      : Factor w/ 2 levels "NO","YES": 1 2 1 1 1 1 1 1 1 1 ...
##  $ pep           : Factor w/ 2 levels "NO","YES": 2 1 1 1 1 2 2 1 1 1 ...
##  $ income.bracket: Factor w/ 3 levels "low","med","high": 1 2 1 1 3 2 1 2 2 1 ...
##  $ age.group     : Factor w/ 3 levels "(18,34.3]","(34.3,50.7]",..: 2 2 3 1 3 3 1 3 2 3 ...
```

### Performing Association Rule Mining:  

```r
rule1 <- apriori(data = data, parameter = list(minlen=2, maxlen=100, supp = 0.05, conf=0.8), appearance = list(default="lhs", rhs=c("pep=NO", "pep=YES")))
```

```
## Apriori
## 
## Parameter specification:
##  confidence minval smax arem  aval originalSupport maxtime support minlen
##         0.8    0.1    1 none FALSE            TRUE       5    0.05      2
##  maxlen target   ext
##     100  rules FALSE
## 
## Algorithmic control:
##  filter tree heap memopt load sort verbose
##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
## 
## Absolute minimum support count: 30 
## 
## set item appearances ...[2 item(s)] done [0.00s].
## set transactions ...[28 item(s), 600 transaction(s)] done [0.00s].
## sorting and recoding items ... [28 item(s)] done [0.00s].
## creating transaction tree ... done [0.00s].
## checking subsets of size 1 2 3 4 5 6 7 done [0.01s].
## writing ... [167 rule(s)] done [0.00s].
## creating S4 object  ... done [0.00s].
```

```r
inspect(head(sort(rule1, by="lift", decreasing = T), 5))
```

```
##     lhs                        rhs          support confidence     lift count
## [1] {children=1,                                                             
##      current_act=YES,                                                        
##      age.group=(34.3,50.7]} => {pep=YES} 0.05833333  1.0000000 2.189781    35
## [2] {married=NO,                                                             
##      children=0,                                                             
##      save_act=YES,                                                           
##      mortgage=NO}           => {pep=YES} 0.05166667  0.9687500 2.121350    31
## [3] {married=YES,                                                            
##      children=1,                                                             
##      age.group=(34.3,50.7]} => {pep=YES} 0.05000000  0.9677419 2.119143    30
## [4] {children=1,                                                             
##      mortgage=NO,                                                            
##      income.bracket=med}    => {pep=YES} 0.05000000  0.9677419 2.119143    30
## [5] {children=1,                                                             
##      age.group=(34.3,50.7]} => {pep=YES} 0.07666667  0.9583333 2.098540    46
```

### Findings and Conclusion:  
Below are the five findings from the association rule data mining. 

#### #1.  

If the customer age is between 34 and 50, has a child and also holds the current account, then it is most likely they will purchase PEP product.  

   *Support: 0.0583, Confidence:1.000, Lift: 2.1897*
   
   This pattern doesn't occur more often (Support: 0.0583), but has a strong relationship (Confidence: 1.00) with highest predictive power (Lift: 2.18)
   
   **Recommendation:** Since the occurance of this rule or pattern is fairly low, we will need to concentrate on this demographics. Our target market will need to be other companies where this age group employees can be found. Our marketing has to be more approachable for these demographics.
    
    
#### #2. 

```r
rule2 <- apriori(data = data, parameter = list(minlen=1, maxlen=100, supp = 0.1, conf=0.9), appearance = list(default="lhs", rhs=c("pep=NO", "pep=YES")))
```

```
## Apriori
## 
## Parameter specification:
##  confidence minval smax arem  aval originalSupport maxtime support minlen
##         0.9    0.1    1 none FALSE            TRUE       5     0.1      1
##  maxlen target   ext
##     100  rules FALSE
## 
## Algorithmic control:
##  filter tree heap memopt load sort verbose
##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
## 
## Absolute minimum support count: 60 
## 
## set item appearances ...[2 item(s)] done [0.00s].
## set transactions ...[28 item(s), 600 transaction(s)] done [0.00s].
## sorting and recoding items ... [28 item(s)] done [0.00s].
## creating transaction tree ... done [0.00s].
## checking subsets of size 1 2 3 4 5 done [0.00s].
## writing ... [4 rule(s)] done [0.00s].
## creating S4 object  ... done [0.00s].
```

```r
inspect(head(sort(rule2, by="lift", decreasing = T), 5))
```

```
##     lhs                  rhs        support confidence     lift count
## [1] {married=YES,                                                    
##      children=0,                                                     
##      save_act=YES,                                                   
##      current_act=YES} => {pep=NO} 0.1333333  0.9195402 1.692405    80
## [2] {married=YES,                                                    
##      children=0,                                                     
##      save_act=YES,                                                   
##      mortgage=NO}     => {pep=NO} 0.1216667  0.9125000 1.679448    73
## [3] {married=YES,                                                    
##      children=0,                                                     
##      current_act=YES,                                                
##      mortgage=NO}     => {pep=NO} 0.1333333  0.9090909 1.673173    80
## [4] {sex=FEMALE,                                                     
##      married=YES,                                                    
##      children=0,                                                     
##      mortgage=NO}     => {pep=NO} 0.1050000  0.9000000 1.656442    63
```

If the customer is married, with no children and have Savings and current account, they are most likely not to purchase PEP product. 

  *Support: 0.1333, Confidence: 0.9195, Lift: 1.69*
  
  There is a very strong relationship (~92%) and fairly good possibility of occurence (13%).
  
  **Recommendation:** It is always not important to concentrate on the customers who are intrested in ‘purchasing’ the product, but also get demographics data of customers that are “NOT” purchasing the product. Hence from this rule we can get some indication on that. We need to increase advertising or awareness among this spectrum of customers, in order to increase our product sales.

#### #3. 

If the customer is married, No children, having Savings Account and No mortgage, are more likely NOT to purchase a PEP product.

  *Support: 0.1216, Confidence: 0.9125, Lift: 1.679*
  
  From the above pattern, we see that there is a very strong relationship (91%) and fairly good possibility of occurance (12%) when we consider the demographics in this pattern. The reason could be that the customers are newly married or have been married for a while, but are not yet ready to take huge investments.
  
  **Recommendation:** We will need to increase awareness and educate customers of this nature about - the need to invest in equities, the profits and benefits for their future. This will help us target that whole demographics of customers who have not purchased the PEP product yet, but will start doing it.
  
#### #4. 

If the customer is a Married Female with No children and No mortgage, then also it is more likely Not to purchase a PEP product.

  *Support: 0.105, Confidence: 0.900, Lift: 1.656*
  
  This pattern has a good and strong relationship (90%) and definetly a fairly good possibility of occurance (10%), with a fair predictive power (i.e. 1.6 when compared to highest predictive power of 2.18).
  
  **Recommendation:** Increase the ease of understanding and target all gender crowd to purchase the product. Since this demographics mentions that the female customer is married, it can also advertise or pursuade their husband to purchase the product.

#### #5. 


```r
rule3 <- apriori(data = data, parameter = list(minlen=1, maxlen=100, supp = 0.05, conf=0.8), appearance = list(default="lhs", rhs=c("pep=NO", "pep=YES")))
```

```
## Apriori
## 
## Parameter specification:
##  confidence minval smax arem  aval originalSupport maxtime support minlen
##         0.8    0.1    1 none FALSE            TRUE       5    0.05      1
##  maxlen target   ext
##     100  rules FALSE
## 
## Algorithmic control:
##  filter tree heap memopt load sort verbose
##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
## 
## Absolute minimum support count: 30 
## 
## set item appearances ...[2 item(s)] done [0.00s].
## set transactions ...[28 item(s), 600 transaction(s)] done [0.00s].
## sorting and recoding items ... [28 item(s)] done [0.00s].
## creating transaction tree ... done [0.00s].
## checking subsets of size 1 2 3 4 5 6 7 done [0.01s].
## writing ... [167 rule(s)] done [0.00s].
## creating S4 object  ... done [0.00s].
```

```r
inspect(head(sort(rule3, by="count", decreasing = T), 5))
```

```
##     lhs                                      rhs       support  
## [1] {children=1}                          => {pep=YES} 0.1833333
## [2] {married=YES,children=0,save_act=YES} => {pep=NO}  0.1783333
## [3] {married=YES,children=0,mortgage=NO}  => {pep=NO}  0.1733333
## [4] {children=1,current_act=YES}          => {pep=YES} 0.1400000
## [5] {children=1,save_act=YES}             => {pep=YES} 0.1333333
##     confidence lift     count
## [1] 0.8148148  1.784266 110  
## [2] 0.8991597  1.654895 107  
## [3] 0.8965517  1.650095 104  
## [4] 0.8316832  1.821204  84  
## [5] 0.8421053  1.844026  80
```

  There are pretty high number of customers, who are likely to purchase the PEP product when they have a child and also a current or savings account.
  
  {children=1,current_act=YES}
  {children=1,save_act=YES}

  *Supprt= 0.1400000 Confidence=0.83 Lift=1.821204*

  Both above patterns have similar association. We can see that these patterns, where the customer has a child and has either current account or savings account have a more likely tendency to buy the PEP product. As, the attributes have a strong relationship (~84%) and fair probability of occurance (~14%) with a preditictive power of about 1.8.
  
  **Recommendation:** It is good to concentrate on these demographics and try to increase sales marketing and promoting the product for customers with bank accounts. They seem to have a more likely nature to purchase the PEP product and have the money for investments.
  

### References:
1. Apriori documentation.
2. Arules documentation.
