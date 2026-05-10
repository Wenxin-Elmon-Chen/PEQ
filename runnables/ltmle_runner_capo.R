suppressPackageStartupMessages(library(ltmle))
suppressPackageStartupMessages(library(SuperLearner))
suppressPackageStartupMessages(library(arm))
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(randomForest))


# Create dataframe for ltmle input
# Expect `data` shaped (n, T, p) with column order: [Y, A, X...]
create_data_input <- function(data){
  n = dim(data)[1]
  T = dim(data)[2]
  p = dim(data)[3]
  L_nodes <- c()
  A_nodes <- c()
  Y_nodes <- c()
  data_tmle <- data.frame(matrix(ncol = 0, nrow = n))
  for (t in 1:T){
    # Covariates
    X_t <- data[,t,3:p]
    for (i in 1:(p-2)){
      name_X <- paste("X_",t,sep="")
      name_X <- paste(name_X,"_",sep="")
      name_X <- paste(name_X,i,sep="")
      data_tmle[name_X] <- X_t[,i]
      # Baseline covariates are not included in L_nodes
      if (t > 1){
        L_nodes <- c(L_nodes,name_X)
      }
    }
    # Treatment: convert to integer
    A_t <- data[,t,2]
    name_A <- paste("A_",t,sep="")
    data_tmle[name_A] <- as.integer(A_t)
    A_nodes <- c(A_nodes,name_A)
  }
  # No censoring in this simulator: omit censoring nodes entirely.
  # Note: creating a constant factor censoring column (e.g., "uncensored") causes
  # some learners (notably xgboost) to error because the label is not a binary outcome.
  C_nodes <- NULL
  # Outcome
  Y <- data[,T,1]
  data_tmle["Y"] <- as.numeric(Y)
  Y_nodes <- "Y"

  return(list(data_tmle = data_tmle, L_nodes = L_nodes, A_nodes = A_nodes, Y_nodes = Y_nodes,
              C_nodes = C_nodes))
}

# Public API for Python / rpy2.
#
# Returns APO = E[Y^{a_int}] (the "treatment" regimen mean), not the ATE.
#
# - data: numeric array (n, T, p), where p >= 3, with [Y, A, X...]
# - a_int: length-T vector of 0/1 specifying intervention sequence
# - method:
#     * "ltmle_super": standard LTMLE with SL libraries for Q and g
#     * "iptw_only": IPTW-only diagnostic using only the g model
#     * "gcomp": g-computation (ltmle's gcomp=TRUE)
estimate_apo <- function(data,
                         a_int,
                         method = c("ltmle_super", "iptw_only", "gcomp"),
                         v_folds = 3) {
  method <- match.arg(method)
  T <- dim(data)[2]
  if (length(a_int) != T) stop("a_int must have length T")

  ltmle_list <- create_data_input(data)
  abar <- as.numeric(a_int)

  # User-requested library: glm, rf, xgboost, gam
  SL.lib <- c("SL.glm", "SL.randomForest", "SL.xgboost", "SL.gam")

  if (method == "ltmle_super") {
    result <- suppressWarnings(suppressMessages(ltmle(ltmle_list$data_tmle,
                                                      Anodes = ltmle_list$A_nodes,
                                                      Lnodes = ltmle_list$L_nodes,
                                                      Ynodes = ltmle_list$Y_nodes,
                                                      Cnodes = ltmle_list$C_nodes,
                                                      abar = abar,
                                                      SL.library = list(Q = SL.lib, g = SL.lib),
                                                      SL.cvControl = list(V = v_folds),
                                                      estimate.time = TRUE)))
  }

  if (method == "iptw_only") {
    result <- suppressWarnings(suppressMessages(ltmle(ltmle_list$data_tmle,
                                                      Anodes = ltmle_list$A_nodes,
                                                      Lnodes = ltmle_list$L_nodes,
                                                      Ynodes = ltmle_list$Y_nodes,
                                                      Cnodes = ltmle_list$C_nodes,
                                                      abar = abar,
                                                      SL.library = "SL.glm",
                                                      SL.cvControl = list(V = v_folds),
                                                      variance.method = "ic",
                                                      estimate.time = TRUE,
                                                      iptw.only = TRUE)))
  }

  if (method == "gcomp") {
    result <- suppressWarnings(suppressMessages(ltmle(ltmle_list$data_tmle,
                                                      Anodes = ltmle_list$A_nodes,
                                                      Lnodes = ltmle_list$L_nodes,
                                                      Ynodes = ltmle_list$Y_nodes,
                                                      Cnodes = ltmle_list$C_nodes,
                                                      abar = abar,
                                                      SL.library = list(Q = SL.lib, g = SL.lib),
                                                      SL.cvControl = list(V = v_folds),
                                                      estimate.time = TRUE,
                                                      gcomp = TRUE)))
  }

  apo_est <- result$estimates
  return(apo_est)
}


