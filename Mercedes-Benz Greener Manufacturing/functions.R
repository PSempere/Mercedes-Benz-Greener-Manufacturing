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
    #devuelve la puntuacion 
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

fit_model_ft <- function(paramlist) {
    #if (length(i) > 1L)
    #    return(lapply(i, fit_model_ft))

    current_numTrees <- paramlist[["numTrees"]]
    current_numLeaves <- paramlist[["numLeaves"]]
    current_learningRate <- paramlist[["learningRate"]]
    current_minSplit <- paramlist[["minSplit"]]
    current_numBins <- paramlist[["numBins"]]

    m <- rxFastTrees(formula = form, data = train, type = "regression",
        numTrees = current_numTrees, numLeaves = current_numLeaves, learningRate = current_learningRate,
        minSplit = current_minSplit, numBins = current_numBins)

    scores <- rxPredict(m, test, extraVarsToWrite = names(test))

    sc <- computeR2_silent(actual_vector = scores$y, preds_vector = scores$Score)

    if (sc > best_r2) {
        best_r2 <- sc
        bestParams <- paramlist
    }

    ret <- as.data.frame(sc, bestParams)

    return(ret)
}

fit_model_ft_Bayesian <- function(form, data_to_fit)
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