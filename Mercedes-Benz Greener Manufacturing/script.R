library(MicrosoftML)
library(MicrosoftR)
library(xgboost)
library(MLmetrics)
library(e1071)
library(caret)
#cargamos fichero functions
wd <- getwd()

source(paste(wd, "/functions.r", sep = ""))

#leemos dato
train_source <- read.csv(file = "D:/Data/Mercedes Benz/train.csv")
submission_core <- read.csv(file = "D:/Data/Mercedes Benz/test.csv")

label = "y"

label_train <- train_source$y
id_train <- train_source$ID

#70% para entrenar
threshold <- round(0.7 * nrow(train_source))

set.seed(1234)

#division aleatoria
train_indicator <- sample(1:nrow(train_source), size = threshold)

train <- train_source[train_indicator,]
test <- train_source[-train_indicator,]

#division secuencial
#train <- train_source[1:threshold,]
#test <- train_source[(threshold + 1):nrow(train_source),]

#quitamos para que no afecte al entrenamiento, es autonumerica
train$ID = NULL
test$ID = NULL
#train$y = NULL

features <- names(train)

#preprocesado
for (f in features) {

    #quitar columnas con un solo valor
    if (length(unique(train[[f]])) == 1) {
        train[[f]] = NULL
        submission_core[[f]] = NULL
        test[[f]] = NULL
    }

    #COMENTADO, A FASTTREES LE VA MEJOR CON FACTORES 
    #convertir factores como ints
    #if (is.factor(train[[f]])) {
        #levels = sort(unique(train[[f]]))
        #train[[f]] = as.integer(factor(train[[f]], levels = levels))
        #submission_core[[f]] = as.integer(factor(submission_core[[f]], levels = levels))
        #test[[f]] = as.integer(factor(test[[f]], levels = levels))
    #}

    #convertir characters como ints
    if (is.character(train[[f]])) {
        levels = sort(unique(train[[f]]))
        train[[f]] = as.integer(factor(train[[f]], levels = levels))
        submission_core[[f]] = as.integer(factor(submission_core[[f]], levels = levels))
        test[[f]] = as.integer(factor(test[[f]], levels = levels))
    }
}


#refrescar las features supervivientes
features <- names(train)
#quitamos Y
features <- features[-1]

#construir la formula
form <- paste("y~", paste(features, collapse = "+"), sep = "")

#entrenamiento con valores por defecto
ft <- rxFastTrees(formula = form, data = train, type = "regression", verbose = 0)

grid <- NULL

fit_model_ft(form, data = train)

#puntuar
scores <- rxPredict(ft, test, #suffix = ".rxFastTrees",
                      extraVarsToWrite = names(test)
                      )

computeR2('FastTrees', actual_vector = scores$y, preds_vector = scores$Score)

#rxDForest() 
DForest_model <- rxDForest(formula = form,
                           data = train,
                           seed = 1024,
                           cp = 0.001,
                           nTree = 200,
                           mTry = 2,
                           overwrite = TRUE,
                           reportProgress = 0)
#DForest_model
#class(DForest_model) #"rxDForest" 

scores <- rxPredict(DForest_model, test, #suffix = ".rxDForest",
                      extraVarsToWrite = names(test))

computeR2('DForest', actual_vector = scores$y, preds_vector = scores$y_Pred)

################################################################################
## Boosted tree modeling
################################################################################
BoostedTree_model = rxBTrees(formula = form,
                             data = train,
                             maxDepth = 6,
                             learningRate = 0.1,
                             minSplit = 2,
                             #minBucket = 5,
                             #sampRate = 0.9,
                             nTree = 100,
                             lossFunction = "gaussian",
                             reportProgress = 0)
#BoostedTree_model
#class(BoostedTree_model)

scores <- rxPredict(BoostedTree_model, test, #suffix = ".rxBTrees",
                      extraVarsToWrite = names(test))

computeR2('RxBoostedTrees', actual_vector = scores$y, preds_vector = scores$y_Pred)

################################################################################
## Decision Tree Modelling
################################################################################

#rxDTree
DTree_model = rxDTree(formula = form,
                      data = train,
                      maxDepth = 6,
                      minSplit = 3,
                      minBucket = 3,
                      nTree = 200,
                      computeContext = "RxLocalParallel",
                      reportProgress = 0)

scores <- rxPredict(DTree_model, test, #suffix = ".rxDTree",
                      extraVarsToWrite = names(test))

computeR2('rxDTree', actual_vector = scores$y, preds_vector = scores$y_Pred)

##XGBOOST
#train_xgb_df <- data.frame(train)
#test_xgb_df <- data.frame(test)

#xgb_test_y <- lapply(test_xgb_df$y, as.numeric)
#xgb_train_y <- lapply(train_xgb_df$y, as.numeric)

#train_xgb_df$y <- 0.0

#train_xgb <- xgb.DMatrix(data.matrix(train_xgb_df), label = t(xgb_train_y), missing = NaN)
#test_xgb <- xgb.DMatrix(data.matrix(train_xgb_df), label = t(train_xgb_df$y), missing = NaN)

##Hypertuning?
#xgb_model <- xgb.train(data = train_xgb, nrounds = 2000,
                 ##nfold = 6,
                 #early_stop_round = 10,
                 #objective = "reg:linear",
                 #print_every_n = 10,
                 ##num_class = 38,
                 ##verbose = 1,
                 ##feval = xg_eval_mae,
                 ##eval = mae,
                 #maximize = FALSE)

#xgb_scores <- predict(xgb_model, as.matrix(test_xgb_df))

##puntuar R2 con XGBoost
#r2_xgb <- R2_Score(y_pred = xgb_scores, y_true = test$y)
##mostrar la puntuacion con XGBoost
#r2_xgb