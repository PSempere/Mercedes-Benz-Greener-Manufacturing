library(e1071)
library(caret)
library(rBayesianOptimization)

computeR2 <- function(alg_name, actual_vector, preds_vector)
{
    #puntuar R2
    r2 <- R2_Score(y_pred = preds_vector, y_true = actual_vector)
    #mostrar la puntuacion 
    res <- paste(alg_name, 'R2 Score:', r2)
    return(res)
}

computeR2_silent <- function(actual_vector, preds_vector) {
    #puntuar R2
    r2 <- R2_Score(y_pred = preds_vector, y_true = actual_vector)
    #mostrar la puntuacion 
    res <- paste(alg_name, 'R2 Score:', r2)
    return(r2)
}

fasttree_func <- function(numLeaves, learningRate, minSplit)
{
    ft <- rxFastTrees(formula = form, data = train, type = "regression", verbose = 0,
        numLeaves = numLeaves,
        learningRate = learningRate,
        minSplit = minSplit)

    scores <- rxPredict(ft, test,
                      extraVarsToWrite = names(test)
                      )
    list(Score = computeR2_silent(scores$y, scores$Score), Pred = scores$Score)
}

fit_model_ft <- function(form, data_to_fit)
{
    upperBounds <- c(
    numLeaves = 100,
    learningRate = 0.4,
    minSplit = 20,
    numBins = 2048
    )

    lowerBounds <- c(
    numLeaves = 5,
    learningRate = 0.001,
    minSplit = 2,
    numBins = 16
    )

    bounds <- list(
        numLeaves = c(lowerBounds[1], upperBounds[1]),
        learningRate = c(lowerBounds[2], upperBounds[2]),
        minSplit = c(lowerBounds[3], upperBounds[3])
        #,numBins = c(lowerBounds[4], upperBounds[4])
        )

    set.seed(1234)

    #hypertune!

    bayes_search <- BayesianOptimization(
                        fasttree_func,
                        bounds = bounds,
                        init_points = 0,
                        n_iter = 5,
                        kappa = 1
    )
    
    return(bayes_search)
}