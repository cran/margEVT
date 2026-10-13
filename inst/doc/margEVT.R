## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>", fig.width = 6,
                      fig.height = 4, dpi = 72)

## -----------------------------------------------------------------------------
library(margEVT)

## ----data---------------------------------------------------------------------
set.seed(2024)
n_years <- 40; n_y <- 52
n <- n_years * n_y
week <- rep(seq_len(n_y), n_years)
df <- cbind(
  data.frame(year = rep(1981:2020, each = n_y),
             x = as.numeric(arima.sim(list(ar = 0.8), n)) * 0.6,
             z = rnorm(n)),
  seasonal_harmonics(week, period = n_y)
)
df$y <- 20 + 3 * df$x + 4 * df$cos1 + 3 * rexp(n)
u <- quantile(df$y, 0.9)

## ----fit----------------------------------------------------------------------
fit <- fit_nhpp(df, threshold = u,
                loc_vars   = c("cos1", "sen1", "x", "z"),
                scale_vars = c("x", "z"),
                free_vars  = c("cos1", "sen1"),
                penalty = "lasso", lambda = "bic",
                penalty_weights = "adaptive",
                obs_per_year = n_y, verbose = FALSE)
fit

## ----summary------------------------------------------------------------------
summary(fit)
active_covariates(fit)

## ----path, fig.height = 6-----------------------------------------------------
head(fit$path$table)
plot(fit, type = "path")

## ----qq, fig.width = 7, fig.height = 3.5--------------------------------------
op <- par(mfrow = c(1, 2))
plot(fit, type = "qq", residuals = "interarrival")
plot(fit, type = "qq", residuals = "magnitude")
par(op)

## ----generics-----------------------------------------------------------------
logLik(fit)
predict(fit, newdata = data.frame(cos1 = 1, sen1 = 0, x = c(-1, 0, 1),
                                  z = 0))

## ----generator----------------------------------------------------------------
gen <- fit_var_generator(fit, df, vars = "x", lag_max = 4)
gen$p_opt
sims <- simulate_covariates(gen, n_mc = 300, seed = 1)

## ----marginalize--------------------------------------------------------------
rl <- marginalize(fit, df, TRs = c(2, 10, 50, 100),
                  approaches = c("A", "B", "C"),
                  scenarios = list(neutral = list(x = 0),
                                   positive = list(x = 1)),
                  mc_sample = sims, n_boot = 300, seed = 1)
rl_table(rl)
plot(rl)

## ----bootstrap----------------------------------------------------------------
ci <- bootstrap_rl(fit, df, TRs = c(10, 50), R = 20, approach = "C",
                   marginalize_args = list(n_boot = 100, seed = 1),
                   verbose = FALSE)
ci

## ----backtest-----------------------------------------------------------------
bt <- backtest(fit, df, varname = "y", TRs = c(2, 5),
               window_years = 10, min_train_years = 20, n_boot = 100,
               verbose = FALSE)
bt$binom_tests

## ----session------------------------------------------------------------------
sessionInfo()$R.version$version.string
packageVersion("margEVT")

