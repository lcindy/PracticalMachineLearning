Practical Machine Learning Project 
Project: Predict the manner of exercise
========================================================
##  Background::
The goal is to use data from accerlerometers on the belt, forearm, arm and dumbbell of 6 participants.  They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  This project is to use the data to predict the manner in which the 6 participants did the exercise.  The five behaviours are described as the following:  
A.Exact  
B.Throw elbow to the front  
C.Lift dumbbell half way  
D.Lower dumbbell half way  
E.Throw the hip to the front  

The data is extracted from the following website
http://groupware.les.inf.puc-rio.br/har

##  Data Analysis
### Explore the data:
The data set includes 52 variables as the following:  
1.Roll/pitch/yaw of belt/arm/forearm/dumbbell  
2.Total acceleration of  belt/arm/forearm/dumbbell  
3.Gyros/accel/magnet of belt/arm/forearm/dumbbell in x/y/z  
To clean the dataset, I first filter out the unnecessary columns and include new window = no.   


Then I partition the dataset into training set and cross validation set in the following percentage:    
Training: 70%  
Cross Validation: 30%  


```r
inTrain = createDataPartition(training_data$classe, p = 0.7)[[1]]
training = training_data[ inTrain,]
cv = training_data[-inTrain,]
#number of data points in training
dim(training)
```

```
## [1] 13453    53
```

```r
#number of data points in cross validation 
dim(cv)
```

```
## [1] 5763   53
```
### Fitting the Model:
Random forest is chosen as the main algorithm for training.  The dataset has more than 16000 rows with 52 variables.  Random forest is well known for its accuracy, its ability to work with large data sets and to handle large set of variables.  In addition, it provides accurate information about the importantce of each variables.  Although it is a bit time consuming, I think it's the best model for this project.


```r
set.seed(33833)
#modfit <- train(classe ~ . , data = training, method="rf",prox=TRUE)
library(randomForest)
modfit<-randomForest(classe ~ . , data = training, importance = T)
modfit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, importance = T) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.48%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3824    5    1    0    0    0.001567
## B   13 2583    7    0    0    0.007683
## C    0   10 2336    1    0    0.004687
## D    0    0   20 2180    3    0.010440
## E    0    0    1    3 2466    0.001619
```

```r
plot(modfit)
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 
  
I train the training set using randomForest package.  The resulting average error is 0.48%.  Based on the graph which plots the error rate for 500 trees.  The max error rate is around 15% while most trees have very small error rate.  Given that the average error is less than 0.48% and it is trained with a dataset with 13000 data points.  I expect the out of sample error rate to be potentiall small.  

In addition, by examining the confusion matrix.  Class D (lower dumbbell half way) has the largest error rate of 1%.  Class A (Exact) and Class E(throw the hip to the front) have the lowest error rate.  To further analyze the accuracy of the model, I also look into the variable importance.    


```r
variable_imp<-varImp(modfit)
#Class A
order.A<-order(variable_imp$A,decreasing=TRUE)
Top.A<-head(variable_imp[order.A,],8)
cbind(row.names(Top.A),Top.A$A)
```

```
##      [,1]                [,2]              
## [1,] "yaw_belt"          "47.2162756407411"
## [2,] "magnet_dumbbell_z" "41.9057087332114"
## [3,] "roll_belt"         "40.9511870900922"
## [4,] "pitch_forearm"     "33.0004229141963"
## [5,] "magnet_dumbbell_y" "32.8365776344424"
## [6,] "pitch_belt"        "31.2383011665577"
## [7,] "roll_forearm"      "29.6961173061275"
## [8,] "accel_dumbbell_y"  "26.9096687274176"
```

The most important variables to identify Class A(Exact) are yaw_belt, magnet_dumbbell_z and roll_belt.  I have extracted the top 8 variables.  The sensor around the waist (in the belt area) seems to be the best way to identify the correct posture for barbell lift.  The movements in the forearm and dumbbell also contribute highly to identify Class A.

```r
#Class B
order.B<-order(variable_imp$B,decreasing=TRUE)
Top.B<-head(variable_imp[order.B,],5)
cbind(row.names(Top.B),Top.B$B)
```

```
##      [,1]                [,2]              
## [1,] "pitch_belt"        "51.6166928587082"
## [2,] "roll_belt"         "46.2987656886925"
## [3,] "yaw_belt"          "45.2276462375896"
## [4,] "magnet_dumbbell_z" "41.3812206608155"
## [5,] "magnet_dumbbell_y" "36.9734867219045"
```
The most important variables to identify Class B(Throw Elbow to the front) are pitch_belt, roll_belt and yaw_belt.  It is clearly seen that the movements on the waist (around the belt) play an important role to identify Class B.  


```r
#Class C
order.C<-order(variable_imp$C,decreasing=TRUE)
Top.C<-head(variable_imp[order.C,],5)
cbind(row.names(Top.C),Top.C$C)
```

```
##      [,1]                [,2]              
## [1,] "magnet_dumbbell_z" "47.5488480399765"
## [2,] "magnet_dumbbell_y" "45.4808908230001"
## [3,] "roll_belt"         "43.5299009177209"
## [4,] "yaw_belt"          "41.1341810457893"
## [5,] "pitch_forearm"     "39.8094792971592"
```
Class C is "lift dumbbell half way".  magnet_dumbbell_z and magnet_dumbbell_y are the most important variables to identify Class C.  It is anticipated to see the movements in the dumbbell are most significant.

```r
#Class D
order.D<-order(variable_imp$D,decreasing=TRUE)
Top.D<-head(variable_imp[order.D,],5)
cbind(row.names(Top.D),Top.D$D)
```

```
##      [,1]                [,2]              
## [1,] "yaw_belt"          "46.4438071512378"
## [2,] "roll_belt"         "45.3981175667561"
## [3,] "magnet_dumbbell_z" "37.7611456440204"
## [4,] "magnet_dumbbell_y" "35.9723608951396"
## [5,] "pitch_forearm"     "34.4257796827088"
```
Class D is "lower dumbbell half way".  It is expected to see dumbbell related variables have high importance.  However, it is very interesting to see the movements around the waist are significant.  

```r
#Class E
order.E<-order(variable_imp$E,decreasing=TRUE)
Top.E<-head(variable_imp[order.E,],5)
cbind(row.names(Top.E),Top.E$E)
```

```
##      [,1]                [,2]              
## [1,] "roll_belt"         "43.3907615558338"
## [2,] "magnet_dumbbell_z" "37.451231311117" 
## [3,] "yaw_belt"          "33.3273100740776"
## [4,] "magnet_dumbbell_y" "33.1466666145881"
## [5,] "pitch_forearm"     "33.1453267186269"
```
Class E is "throwing the hip to the front".  It is anticipated that the movement around the waist area (the sensor around the belt) has outpaced the other variables and is most sigficant.  

After analyzing the importance of each variable for all five classes, along with the error rate of 0.48%.  I am confident that random forest model can be a good prediction for this exercise.  Thus, small out of sample error is anticipated.  

### Prediction/Out of Sample Test:
I have test the model with the cross validation set, which is 30% of the training data set. 

```r
pred<-predict(modfit,cv)
table(pred,cv$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1641    8    0    0    0
##    B    0 1105    8    0    0
##    C    0    2  996    9    0
##    D    0    0    1  932    2
##    E    0    0    0    3 1056
```

```r
predRight<-pred==cv$classe
#Total of accurate prediction for out of sample data
accuracy<-sum(predRight)/length(predRight)
qplot(1:length(pred),pred, colour=predRight)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9.png) 
  
The out of sample error rate  is 0.5726% which is very close to the sample error rate of 0.48% from the training set.  We can see from the plot that orange dots are the ones that I failed to predict.  

### Test Results:
Please see below for the prediction of the test class.

```r
test_classe<-predict(modfit,testing_data)
test_classe
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
  
### Conclusion:
The random forest model turns out to be a good model for predicting the correctness of barbell lifts.  The error rate is 0.48%.  By examing the importance of the variables, I am able to look into the contribution of each variable for each class.  When I implement the model with cross validation set, the out of sample error is 0.5726% which is very satisfactory.
