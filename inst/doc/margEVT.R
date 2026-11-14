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

## ----ei-----------------------------------------------------------------------
extremal_index(df$y, u)

## ----decluster----------------------------------------------------------------
df$y_raw <- df$y
df$y <- decluster(df$y_raw, u)
c(exceedances = sum(df$y_raw > u), clusters = sum(df$y > u))

## ----fit----------------------------------------------------------------------
fit <- fit_nhpp(df, threshold = u,
                loc_vars   = c("cos1", "sen1", "x", "z"),
                scale_vars = c("x", "z"),
                free_vars  = c("cos1", "sen1"),
                penalty = "lasso", lambda = "bic",
                penalty_weights = "adaptive", refit = TRUE,
                obs_per_year = n_y, verbose = FALSE)
fit

## ----summary------------------------------------------------------------------
summary(fit)
active_covariates(fit)

## ----path, fig.height = 6-----------------------------------------------------
plot(fit, type = "path")

## ----qq, fig.width = 7, fig.height = 3.5--------------------------------------
op <- par(mfrow = c(1, 2))
plot(fit, type = "qq", residuals = "interarrival")
plot(fit, type = "qq", residuals = "magnitude")
par(op)

## ----generics-----------------------------------------------------------------
logLik(fit)
confint(fit)
predict(fit, newdata = data.frame(cos1 = 1, sen1 = 0, x = c(-1, 0, 1),
                                  z = 0))

## ----generator----------------------------------------------------------------
gen <- fit_var_generator(fit, df, vars = "x", lag_max = 4)
gen
sims <- simulate_covariates(gen, n_mc = 300, seed = 1)

## ----marginalize--------------------------------------------------------------
rl <- marginalize(fit, df, TRs = c(2, 10, 50, 100),
                  approaches = c("A", "B", "C"),
                  scenarios = list(neutral = list(x = 0),
                                   positive = list(x = 1)),
                  mc_sample = sims, n_boot = 300, seed = 1)
rl_table(rl)
plot(rl)

## ----horizon------------------------------------------------------------------
years <- unique(df$year)
traj <- replicate(200, lapply(sample(years, 10, replace = TRUE),
                              function(yr) df[df$year == yr, "x", drop = FALSE]),
                  simplify = FALSE)
hr <- horizon_risk(fit, z = c(40, 45, 50), trajectories = traj)
hr
plot(hr)

## ----dll----------------------------------------------------------------------
design_life_level(fit, traj, prob = c(0.5, 0.1))

## ----bootstrap----------------------------------------------------------------
b <- bootstrap_nhpp(fit, df, TRs = c(10, 50), R = 20, approach = "C",
                    marginalize_args = list(n_boot = 100, seed = 1),
                    verbose = FALSE)
b

## ----backtest-----------------------------------------------------------------
bt <- backtest(fit, df, varname = "y_raw", TRs = c(2, 5),
               window_years = 10, min_train_years = 20, n_boot = 100,
               decluster = function(y) decluster(y, u), verbose = FALSE)
bt

## ----session------------------------------------------------------------------
sessionInfo()$R.version$version.string
packageVersion("margEVT")

