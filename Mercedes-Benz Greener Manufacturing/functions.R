library(e1071)
library(caret)
library(rBayesianOptimization)

computeR2 <- function(alg_name = 'Unknown', actual_vector, preds_vector, silent = 1)
{
    #puntuar R2
    r2 <- R2_Score(y_pred = preds_vector, y_true = actual_vector)
    #mostrar la puntuacion
    res <- paste(alg_name, 'R2 Score:', r2)
    if (silent == 1) { return(r2) } else { return(res) }
}

fit_model_parallel <- function(i) {
    temp_data <- readRDS(paste(wd, "/data.RData", sep=""))

    m <- rxFastTrees(formula = form, data = train, type = "regression",
        numTrees = tunegrid$numTrees[i], numLeaves = tunegrid$numLeaves[i], learningRate = tunegrid$learningRate[i],
        minSplit = tunegrid$minSplit[i], numBins = tunegrid$numBins[i], verbose = 0)

    #resultsTuning$numTrees[i] <- tunegrid$numTrees[i]
    #resultsTuning$numLeaves[i] <- tunegrid$numLeaves[i]
    #resultsTuning$learningRate[i] <- tunegrid$learningRate[i]
    #resultsTuning$minSplit[i] <- tunegrid$minSplit[i]
    #resultsTuning$numBins[i] <- tunegrid$numBins[i]

    scores <- rxPredict(m, test, extraVarsToWrite = names(test))

    sc <- computeR2(actual_vector = scores$y, preds_vector = scores$Score, silent = 1)

    tunegrid$r2[i] <- sc

    if (sc > best_r2) { 
        best_r2 <- sc
        bestParams <- list(tunegrid$numTrees[i], tunegrid$numLeaves[i], tunegrid$learningRate[i], tunegrid$minSplit[i], tunegrid$numBins[i])
    }
}

fit_model_ft <- function(i) {
    if (length(i) > 1L)
        return(lapply(i, fit_model_ft))

    #print(names(paramlist))

    current_numTrees <- tunegrid$numTrees[i]
    current_numLeaves <- tunegrid$numLeaves[i]
    current_learningRate <- tunegrid$learningRate[i]
    current_minSplit <- tunegrid$minSplit[i]
    current_numBins <- tunegrid$numBins[i]

    m <- rxFastTrees(formula = form, data = train, type = "regression",
        numTrees = current_numTrees, numLeaves = current_numLeaves, learningRate = current_learningRate,
        minSplit = current_minSplit, numBins = current_numBins)

    scores <- rxPredict(m, test, extraVarsToWrite = names(test))

    sc <- computeR2(actual_vector = scores$y, preds_vector = scores$Score, silent = 1)

    df <- data.frame(current_numTrees, current_numLeaves, current_learningRate, current_minSplit, current_numBins, sc)

    return (df)
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