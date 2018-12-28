install.packages("jpeg")
BiocInstaller::biocLite("EBImage")
install.packages("caret")
install.packages("e1071")
install.packages("ggfortify")
install.packages("wSVM")
install.packages("psych")
install.packages("xgboost")
install.packages("randomForest")
install.packages("dplyr")
install.packages("MLmetrics")
install.packages("class")
install.packages("kknn")
install.packages("naivebayes")
install.packages("keras")
install.packages("yaml")

library(jpeg)
library(EBImage)
library(caret)
library(e1071)
library(ggfortify)
library(wSVM)
library(psych)
library(xgboost)
library(randomForest)
library(dplyr)
library(MLmetrics)
library(class)
library(kknn)
library(naivebayes)
library(keras)
library(yaml)

## directory in which images are stored
imagedir <- "D:\\NCI\\3.semester_3\\1.trash_classification\\DATA\\trashnet-master\\data\\dataset-resized"
categories <- list.dirs(imagedir, full.names = FALSE, recursive = FALSE)
filecount <- 0
maxvectorsize <- 0
filecount <- 0
filecategory <- 0
comp_factor <- 0.1
dimx <- round(512*comp_factor)
dimy <- round(384*comp_factor)
imagemat <- matrix(nrow = filecount, ncol = (dimx*dimy*3)+1)
imagedf <- as.data.frame(imagemat)
rm(imagemat)
gc()
imagedf$V1 <- factor(imagedf$V1,levels = categories)

# Extract the R,G,B values of every pixel and store it in a vector
for (category in categories) {
  #iterate over our categories
  files <- list.files(paste(imagedir,category,sep="\\"),pattern = NULL)
  filecategory <- filecategory+1
  for (file in files) {
    #loop over each image file in directory
    image <- readJPEG(paste(imagedir,category,file,sep="\\"))
    filecount <- filecount+1
    print(paste(filecount, filecategory, file))
    image <- resize(image, dimx, dimy)
    r <- as.vector(image[,,1])
    g <- as.vector(image[,,2])
    b <- as.vector(image[,,3])
    featurevector <- t(c(r,g,b))
    imagedf[filecount,2:length(featurevector)+1] <- featurevector
    imagedf[filecount,1] <- category
    imagedf[filecount,2] <- filecategory
    imagedf[filecount,3] <- file
    imagedf[filecount,4] <- paste(imagedir,category,file,sep="\\")
  }
}

names(imagedf)[1] <- "Category"
names(imagedf)[2] <- "Category_Code"
names(imagedf)[3] <- "File_Name"
names(imagedf)[4] <- "Path"

# TO remove NA, if present.
sum(is.na(imagedf))
colnames(imagedf)[colSums(is.na(imagedf)) > 0]
imagedf <- imagedf[,-(length(imagedf))]

# Remove unwanted columns
imagedf_nocat <- imagedf[,-1]
imagedf_nocat <- imagedf_nocat[,-2]
imagedf_nocat <- imagedf_nocat[,-2]

imagedf_nocat[1:10,1:10]
sum(is.na(imagedf_nocat))

index <- createDataPartition(imagedf_nocat$Category_Code, p = .25, list = FALSE)
train <- imagedf_nocat[-index,]
test <- imagedf_nocat[index,]

table(train$Category_Code)
table(test$Category_Code)

# KMO TEST and Bartletts test
bart_test <- bartlett.test(train[,-1])
kmo_test <- KMO(train[,-1])

# SVM Modelling
svm_fit <- svm(Category_Code ~ ., data=train, kernel = "radial")
svm_pred <- predict(svm_fit, newdata = test)
svm_fit_cm <- confusionMatrix(svm_pred, test$Category_Code)

# Tuning SVM for better accuracy by predicting best values for Cost and Gamma
tune.svm <-
  tune(svm,as.factor(Category_Code) ~.,
       data=train,
       kernel="radial",
       scale=F,
       ranges=list(cost=10^seq(-2,2,length=10), 
                   gamma=10^seq(-2,2)),
       tunecontrol=tune.control(cross=3))

C <- tune.svm$best.parameters[1]
G <- tune.svm$best.parameters[2]
C
G

# Fitting SVM model for train data.
svmfit <- svm(as.factor(Category_Code) ~., 
              data = train,
              kernel = "radial",
              cost =  C,
              gamma = G,
              scale = F)

pred.svm <- predict(svmfit, newdata = test)
mean(pred.svm != test$Category_Code)  #Error Rate
table(pred.svm, test$Category_Code)
svm_confmat <- confusionMatrix(pred.svm, as.factor(test$Category_Code))
svm_confmat

# Random Forest Modelling

rf <- randomForest(as.factor(Category_Code)~., data=train, ntree=5000)
pred <- predict(rf, newdata = test)
rf_err_rt <- mean(pred != test$Category_Code)  #Error rate
table(pred, test$Category_Code)
rf_confmat <- confusionMatrix(pred, as.factor(test$Category_Code))
rf_confmat

###### eXtreme Gradient Boosting algorithm modelling

# Create training and validation sets
train_dat <- imagedf_nocat[-index,]
val_dat <- imagedf_nocat[index,]

# Create numeric labels with one-hot encoding
train_labs <- as.numeric(train_dat$Category_Code) - 1
val_labs <- as.numeric(val_dat$Category_Code) - 1

new_train <- model.matrix(~ . + 0, data = train_dat[, -1])
new_val <- model.matrix(~ . + 0, data = val_dat[, -1])

# Prepare matrices
xgb_train <- xgb.DMatrix(data = new_train, label = train_labs)
xgb_val <- xgb.DMatrix(data = new_val, label = val_labs)

# Set parameters(default)
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 5, eval_metric = "mlogloss")

# these are the datasets the rmse is evaluated for at each iteration
watchlist = list(train=xgb_train, test=xgb_val)

# Calculate # of folds for cross-validation
xgbcv <- xgb.cv(params = params, 
                data = xgb_train, 
                nrounds = 100, # run for 1000 
                nfold = 10, 
                showsd = TRUE, 
                stratified = TRUE, 
                print.every.n = 10, 
                early_stop_round = 20, 
                maximize = FALSE, 
                prediction = TRUE)

# Function to compute classification error
classification_error <- function(conf_mat) {
  conf_mat = as.matrix(conf_mat)
  error = 1 - sum(diag(conf_mat)) / sum(conf_mat)
  return (error)
}

# Mutate xgb output to deliver hard predictions
xgb_train_preds <- data.frame(xgbcv$pred) %>% mutate(max = max.col(., ties.method = "last"), label = train_labs +1)

# Examine output
head(xgb_train_preds)

# Confustion Matrix
### xgb_conf_mat <- table(true = as.numeric(train_y) + 1, pred = xgb_train_preds$max)
xgb_conf_mat <- table(true = as.numeric(train$Category_Code) + 1, pred = xgb_train_preds$max)

# Error 
cat("XGB Training Classification Error Rate:", classification_error(xgb_conf_mat), "\n")
                                                                                                   
xgb_conf_mat_2 <- confusionMatrix(factor(xgb_train_preds$label),
                                  factor(xgb_train_preds$max),
                                  mode = "everything")

print(xgb_conf_mat_2)

# Create the model 
xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 100)

# Predict for validation set
xgb_val_preds <- predict(xgb_model, newdata = xgb_val)

xgb_val_out <- matrix(xgb_val_preds, nrow = 5, ncol = length(xgb_val_preds) / 5) %>% 
  t() %>%
  data.frame() %>%
  mutate(max = max.col(., ties.method = "last"), label = val_labs + 1)

# Confustion Matrix
xgb_val_conf <- table(true = val_labs + 1, pred = xgb_val_out$max)

cat("XGB Validation Classification Error Rate:", classification_error(xgb_val_conf), "\n")

# Automated confusion matrix using "caret"
xgb_val_conf2 <- confusionMatrix(factor(xgb_val_out$label),
                                 factor(xgb_val_out$max),
                                 mode = "everything")

print(xgb_val_conf2)

############ K-Nearest-Neighbour Classification algorithm modelling
# Normalizing the data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) 
}

#Converting the dataset into a normalized form
knn.df<-as.data.frame(lapply(train[,-1], normalize))
knn_train<-knn.df[train,]
knn_test<-knn.df[testsample,]

knn_test_pred <- knn(train = train, test = test, cl = train$Category_Code)
conf_mat_knn <- confusionMatrix(knn_test_pred, as.factor(test$Category_Code))
conf_mat_knn

############ Convolutional Neural Networks
# Read Images
# Reduce the number of images to be loaded in the dataset
index_CNN <- createDataPartition(imagedf$Category_Code, p = .2, list = FALSE)
new_CNN_dataset <- imagedf[index_CNN,]

dim(new_CNN_dataset_test)

index_CNN2 <- createDataPartition(new_CNN_dataset$Category_Code, p = .25, list = FALSE)
new_CNN_dataset_train <- new_CNN_dataset[-index_CNN2,]
new_CNN_dataset_test <- new_CNN_dataset[index_CNN2,]
CNN_train <- new_CNN_dataset_train[,-3:-4]
CNN_test <- new_CNN_dataset_test[,-3:-4]
CNN_train <- CNN_train[,-1]
CNN_test <- CNN_test[,-1]

train_list_path <- new_CNN_dataset_train[,4]
test_list_path <- new_CNN_dataset_test[,4]

train_cat_code <- new_CNN_dataset_train[,2]
test_cat_code <- new_CNN_dataset_test[,2]

train_CNN <- list()
test_CNN <- list()
for (i in 1:dim(new_CNN_dataset_train)[1]) {train_CNN[[i]] <- resize(readImage(train_list_path[i]), dimx, dimy)}
for (i in 1:dim(new_CNN_dataset_test)[1]) {test_CNN[[i]] <- resize(readImage(test_list_path[i]), dimx, dimy)}

train_CNN_new <- train_CNN
test_CNN_new <- test_CNN

gc()

train_CNN_new <- EBImage::combine(train_CNN_new)
test_CNN_new <- EBImage::combine(test_CNN_new)
str(train_CNN_new)

# Reorder dimension
train_CNN_small <- aperm(train_CNN_new, c(4, 1, 2, 3))
test_CNN_small <- aperm(test_CNN_new, c(4, 1, 2, 3))
str(train_CNN_small)

# Response
trainy <- train_cat_code
testy <- test_cat_code

# One hot encoding
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

# Model building
model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(dimx, dimy, 3)) %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 6, activation = 'softmax') %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(lr = 0.01,
                                    decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))
summary(model)

history <- model %>%
  fit(train_CNN_small,
      trainLabels,
      epochs = 60,
      batch_size = 32,
      #      validation_split = 0.2,
      validation_data = list(test_CNN_small, testLabels))
plot(history)

# Capture all confusion matrices in variables to be exported as Exvcel for analysing the performence.
knn_class <- as.matrix(conf_mat_knn, what = "classes")
svm_class <- as.matrix(svm_confmat, what = "classes")
rf_class <- as.matrix(rf_confmat, what = "classes")
xgb_conf_mat_2 <- as.matrix(xgb_conf_mat_2, what = "classes")

write.csv(knn_class, file = "D://knn_class.csv")
write.csv(svm_class, file = "D://svm_class.csv")
write.csv(rf_class, file = "D://rf_class.csv")
write.csv(xgb_conf_mat_2, file = "D://xgb_conf_mat_2.csv")


########################### END OF CODE ####################################