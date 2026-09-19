# ==============================================================================
# Exercise 2 -- R code of Solutions.ipynb (Python + pyblp)
#
# One-time setup in the R console (skip if already done for Exercise 1):
#   install.packages(c("dplyr", "reticulate"))
#   library(reticulate)
#   python <- Sys.which(c("python3", "python"))
#   python <- python[nzchar(python)][1]
#   virtualenv_create("r-pyblp", python = python)
#   virtualenv_install("r-pyblp", c("pyblp", "pandas==2.2.3", "numpy", "statsmodels"))
#
# Run sections interactively, or from the repository root with:
#   source("Exercises/Exercise-2/Solutions.R")
# All six supplemental questions are included and take longer to run.
#
# ==============================================================================

library(dplyr)
library(reticulate)

use_virtualenv("r-pyblp", required = TRUE)
pyblp <- import("pyblp", convert = FALSE)
np <- import("numpy", convert = FALSE)
pyblp$options$digits <- 3L
pyblp$options$verbose <- FALSE

set.seed(0)

# Prefer the repository's data, whether working from the root or Exercise-2.
# Fall back to the same URLs as the notebook for other working directories.
read_exercise_data <- function(filename) {
  paths <- file.path(c("Exercises/Data", "../Data", "Data"), filename)
  available <- paths[file.exists(paths)]
  path <- if (length(available)) available[1] else paste0(
    "https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/",
    filename)
  read.csv(path, stringsAsFactors = FALSE)
}

## ---- 0. Relevant code from Exercise 1 ---------------------------------------

product_data <- read_exercise_data("products.csv") %>%
  mutate(market_size = city_population * 90,
         shares = servings_sold / market_size) %>%
  rename(market_ids = market, product_ids = product,
         prices = price_per_serving)

first_stage <- lm(
  prices ~ 0 + price_instrument + factor(market_ids) + factor(product_ids),
  data = product_data)
# HC0 changes standard errors, but not the fitted values used below.

product_data <- product_data %>%
  rename(demand_instruments0 = price_instrument)

iv_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  product_data)
iv_results <- iv_problem$solve(method = "1s")
print(iv_results)

counterfactual_market <- "C01Q2"
counterfactual_data <- product_data %>%
  filter(market_ids == counterfactual_market) %>%
  select(product_ids, mushy, prices, shares)
counterfactual_data$new_prices <- counterfactual_data$prices
counterfactual_data$new_prices[counterfactual_data$product_ids == "F1B04"] <-
  counterfactual_data$new_prices[counterfactual_data$product_ids == "F1B04"] / 2
counterfactual_data$new_shares <- as.numeric(py_to_r(iv_results$compute_shares(
  market_id = counterfactual_market, prices = counterfactual_data$new_prices)))
counterfactual_data$iv_change <- 100 *
  (counterfactual_data$new_shares - counterfactual_data$shares) /
  counterfactual_data$shares

## ---- 1. Describe cross-market variation -------------------------------------

choice_variation <- product_data %>%
  group_by(market_ids) %>%
  summarise(products = n(),
            mushy_mean = mean(mushy), mushy_std = sd(mushy),
            prices_mean = mean(prices), prices_std = sd(prices),
            .groups = "drop")
print(summary(select(choice_variation, -market_ids)))

# The same 24 products appear in each market (the notebook's "20" is a typo).
# Mushyness does not change across markets, but prices do. This supports
# identifying unobserved price heterogeneity using cross-market variation;
# there is no corresponding choice-set variation for mushy or a constant.

demographic_data <- read_exercise_data("demographics.csv") %>%
  rename(market_ids = market) %>%
  mutate(log_income = log(quarterly_income))
print(demographic_data[sample(nrow(demographic_data), 5), ])
print(summary(select(demographic_data, quarterly_income, log_income)))

demographic_variation <- demographic_data %>%
  group_by(market_ids) %>%
  summarise(log_income_mean = mean(log_income),
            log_income_std = sd(log_income), .groups = "drop")
print(summary(select(demographic_variation, -market_ids)))

# Income has a long right tail, so we work with log income. Its distribution
# varies across markets, providing variation to estimate interactions of
# income with product characteristics. Here we leave out income alone,
# following the notebook's specification with market fixed effects.

## ---- 2. Estimate a parameter on mushy x log income --------------------------

# Draw with replacement within each market using pandas' random_state = 0
sample_demographics <- function(draws) {
  demographics_py <- r_to_py(as.data.frame(
    select(demographic_data, market_ids, log_income)))
  sampled <- demographics_py$groupby("market_ids", as_index = FALSE)$sample(
    n = as.integer(draws), replace = TRUE, random_state = 0L)
  py_to_r(sampled$reset_index(drop = TRUE))
}

agent_data <- sample_demographics(1000L)
nodes <- py_to_r(np$random$default_rng(seed = 0L)$normal(
  size = tuple(as.integer(nrow(agent_data)), 3L)))
agent_data[, c("nodes0", "nodes1", "nodes2")] <- nodes
agent_data$weights <- 1 / 1000
print(agent_data[sample(nrow(agent_data), 5), ])

# left_join preserves the product row order, including alignment with the
# first-stage fitted values and the market-specific counterfactual vectors.
product_data <- product_data %>%
  left_join(select(demographic_variation, market_ids, log_income_mean),
            by = "market_ids") %>%
  mutate(demand_instruments1 = log_income_mean * mushy)

product_formulations <- tuple(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  pyblp$Formulation("0 + mushy"))
agent_formulation <- pyblp$Formulation("0 + log_income")
mushy_problem <- pyblp$Problem(
  product_formulations, product_data, agent_formulation, agent_data)
print(mushy_problem)

optimization <- pyblp$Optimization(
  "trust-constr", list(gtol = 1e-8, xtol = 1e-8))

pyblp$options$verbose <- TRUE
mushy_results <- mushy_problem$solve(
  sigma = 0, pi = 1, method = "1s", optimization = optimization)
pyblp$options$verbose <- FALSE

# Check the objective, first-order conditions, and second-order conditions.
estimation_diagnostics <- function(results) {
  data.frame(
    Converged = py_to_r(results$converged),
    Objective = as.numeric(py_to_r(results$objective)),
    Gradient_Norm = as.numeric(py_to_r(results$projected_gradient_norm)),
    Min_Hessian_Eigenvalue = min(py_to_r(results$reduced_hessian_eigenvalues)))
}
print(estimation_diagnostics(mushy_results))

# Pi is about 0.251: higher-income consumers prefer mushy cereal more.
# WTP changes by Pi / (-alpha) per unit of log income. A 1% increase in
# income changes log income by log(1.01), so multiply by this factor.
# This corrects the notebook's interpretation of the 1% income change.

mushy_wtp_1pct <- as.numeric(py_to_r(mushy_results$pi)) /
  -as.numeric(py_to_r(mushy_results$beta)) * log(1.01)
cat(sprintf("\nExtra WTP for mushy after a 1%% income increase: $%.6f per serving\n",
            mushy_wtp_1pct))

## ---- 3. Check random starting values ----------------------------------------

pi_bounds <- tuple(-10, 10)
starting_results <- vector("list", 3)
for (seed in 0:2) {
  initial_pi <- as.numeric(py_to_r(
    np$random$default_rng(seed = as.integer(seed))$uniform(-10, 10)))
  seed_results <- mushy_problem$solve(
    sigma = 0, pi = initial_pi, pi_bounds = pi_bounds,
    method = "1s", optimization = optimization)
  starting_results[[seed + 1]] <- cbind(
    data.frame(Seed = seed, Initial_Pi = initial_pi,
               Estimated_Pi = as.numeric(py_to_r(seed_results$pi))),
    estimation_diagnostics(seed_results))
}
print(bind_rows(starting_results))

# The estimates agree across starts. A near-zero GMM criterion, which is
# nonnegative, also supports finding the global minimum in this example.

## ---- 4. Evaluate changes to the price cut counterfactual --------------------

counterfactual_data$new_shares <- as.numeric(py_to_r(mushy_results$compute_shares(
  market_id = counterfactual_market, prices = counterfactual_data$new_prices)))
counterfactual_data$mushy_change <- 100 *
  (counterfactual_data$new_shares - counterfactual_data$shares) /
  counterfactual_data$shares
print(counterfactual_data)

# The new interaction is small, so results are close to pure logit. There is
# slightly more substitution from other mushy cereals: these products also
# attract the higher-income consumers who prefer the now-cheaper mushy good.

## ---- 5. Add price x log income and unobserved price heterogeneity -----------

product_data$predicted_prices <- as.numeric(fitted(first_stage))
print(cor(select(product_data, prices, predicted_prices)))

compute_differentiation <- function(x) rowSums(outer(x, x, "-")^2)
product_data <- product_data %>%
  mutate(demand_instruments2 = log_income_mean * predicted_prices) %>%
  group_by(market_ids) %>%
  mutate(demand_instruments3 = compute_differentiation(predicted_prices)) %>%
  ungroup()

product_formulations <- tuple(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  pyblp$Formulation("0 + mushy + prices"))
rc_problem <- pyblp$Problem(
  product_formulations, product_data, agent_formulation, agent_data)
print(rc_problem)

# Rows of Sigma and Pi follow X2: mushy, prices. Pi has one column for income.
# Zeros in Sigma are fixed. Only the price standard deviation is estimated.
# PyBLP uses nodes0 for this single active random coefficient, even though
# the price coefficient is in the second row of X2.
# API: https://pyblp.readthedocs.io/en/stable/_api/pyblp.Problem.solve.html

pyblp$options$verbose <- TRUE
rc_results <- rc_problem$solve(
  sigma = matrix(c(0, 0, 0, 1), nrow = 2, byrow = TRUE),
  pi = matrix(c(0.2, 1), ncol = 1),
  method = "1s", optimization = optimization)
pyblp$options$verbose <- FALSE
print(estimation_diagnostics(rc_results))

rc_sigma <- py_to_r(rc_results$sigma)
rc_pi <- py_to_r(rc_results$pi)
alpha <- as.numeric(py_to_r(rc_results$beta))
mean_log_income <- mean(demographic_data$log_income)
sd_log_income <- sd(demographic_data$log_income)
print(data.frame(
  Mean_Log_Income = mean_log_income,
  Average_Price_Coefficient = alpha + rc_pi[2, 1] * mean_log_income,
  Previous_Price_Coefficient = as.numeric(py_to_r(mushy_results$beta)),
  SD_From_Income = abs(rc_pi[2, 1]) * sd_log_income,
  SD_Unobserved = rc_sigma[2, 2]))

# alpha_it = alpha + sigma_price * nodes0 + pi_price * log_income.
# The average is about -34.5, compared with the previous estimate of -30.6.
# Pi_price is negative: higher-income consumers are more price-sensitive
# in this fitted model. Income and unobserved tastes generate price
# heterogeneity of comparable size (SDs around 5.4 and 6.0, respectively).

## ---- 6. Evaluate changes to the price counterfactual -------------------------

counterfactual_data$new_shares <- as.numeric(py_to_r(rc_results$compute_shares(
  market_id = counterfactual_market, prices = counterfactual_data$new_prices)))
counterfactual_data$rc_change <- 100 *
  (counterfactual_data$new_shares - counterfactual_data$shares) /
  counterfactual_data$shares
print(counterfactual_data)

# The price cut produces a larger increase in demand for F1B04. Substitution
# now varies more along the price dimension: similarly priced cereals tend
# to lose more demand. Unobserved heterogeneity in mushy preferences is still
# missing, because we lack cross-market choice-set variation in mushy.

# ==============================================================================
# Supplemental Questions
# ==============================================================================

## ---- S1. Use different numbers of Monte Carlo draws -------------------------

# Start close to the preceding estimates to speed up the repeated fits.
sigma_start <- matrix(c(0, 0, 0, 6), nrow = 2, byrow = TRUE)
pi_start <- matrix(c(0.2, -6), ncol = 1)

solve_agents <- function(new_specification, new_agent_data) {
  new_problem <- pyblp$Problem(
    product_formulations, product_data, agent_formulation, new_agent_data)
  new_results <- new_problem$solve(
    sigma = sigma_start, pi = pi_start,
    method = "1s", optimization = optimization)
  sigma <- py_to_r(new_results$sigma)
  pi <- py_to_r(new_results$pi)
  data.frame(
    Specification = new_specification,
    Time = as.numeric(py_to_r(new_results$optimization_time)),
    Converged = py_to_r(new_results$converged),
    Gradient_Norm = as.numeric(py_to_r(new_results$projected_gradient_norm)),
    Sigma_on_Price = sigma[2, 2],
    Pi_on_Mushy = pi[1, 1], Pi_on_Price = pi[2, 1],
    Alpha = as.numeric(py_to_r(new_results$beta)))
}

agent_results <- list()
for (draws in c(10L, 100L, 500L, 1000L, 2000L)) {
  cat(sprintf("\nUsing %d Monte Carlo draws per market ...\n", draws))
  mc_data <- sample_demographics(draws)
  mc_data$nodes0 <- as.numeric(py_to_r(np$random$default_rng(seed = 0L)$normal(
    size = as.integer(nrow(mc_data)))))
  mc_data$weights <- 1 / draws
  agent_results[[length(agent_results) + 1]] <- solve_agents(
    paste(draws, "Monte Carlo Draws"), mc_data)
}
print(bind_rows(agent_results))

# Estimates broadly stabilize by 1,000 draws. With few draws, integration
# error can substantially change the estimated preference distribution.
# These one-column normal draws differ from the three-column draws in Q2,
# so even the 1,000-draw result need not exactly match rc_results.

## ---- S2. Use scrambled Halton sequences -------------------------------------

for (draws in c(10L, 100L, 500L, 1000L, 2000L)) {
  cat(sprintf("\nUsing %d Halton draws per market ...\n", draws))
  halton_data <- sample_demographics(draws)
  halton_integration <- pyblp$Integration(
    "halton", size = as.integer(nrow(halton_data)),
    specification_options = list(seed = 0L))
  halton_nodes <- pyblp$build_integration(halton_integration, dimensions = 1L)
  halton_data$nodes0 <- as.numeric(py_to_r(halton_nodes$nodes))
  halton_data$weights <- 1 / draws
  agent_results[[length(agent_results) + 1]] <- solve_agents(
    paste(draws, "Halton Draws"), halton_data)
}
print(bind_rows(agent_results))

# As in the notebook, one long scrambled sequence is split across markets,
# while demographics are still sampled randomly. Halton integration can
# improve precision with fewer nodes; this need not hold for every draw count.

## ---- S3. Try quadrature -----------------------------------------------------

# Seven nodes per dimension give 7^2 = 49 consumer types per market.
# API: https://pyblp.readthedocs.io/en/stable/_api/pyblp.build_integration.html

quad_integration <- pyblp$build_integration(
  pyblp$Integration("product", size = 7L), dimensions = 2L)
quad_nodes <- py_to_r(quad_integration$nodes)
quad_data <- data.frame(
  nodes0 = quad_nodes[, 1], nodes1 = quad_nodes[, 2],
  weights = as.numeric(py_to_r(quad_integration$weights)))
print(quad_data)

# Match the notebook's code: nodes1 becomes log income; nodes0 continues
# to describe unobserved price tastes. Income is now assumed lognormal
# within each market, instead of using its empirical distribution.
quad_data <- merge(quad_data, as.data.frame(demographic_variation),
                   by = NULL, sort = FALSE) %>%
  mutate(log_income = log_income_mean + log_income_std * nodes1) %>%
  select(market_ids, log_income, nodes0, weights)
print(head(quad_data))
agent_results[[length(agent_results) + 1]] <- solve_agents(
  "49 Quadrature Nodes", quad_data)
print(bind_rows(agent_results))

# Results are similar with much less computation. Remaining differences
# reflect both numerical integration and the parametric income assumption.

## ---- S4. Approximate the optimal instruments --------------------------------

optimal_iv_results <- rc_results$compute_optimal_instruments()
optimal_problem <- optimal_iv_results$to_problem()
print(optimal_problem)

pyblp$options$verbose <- TRUE
optimal_results <- optimal_problem$solve(
  sigma = sigma_start, pi = pi_start,
  method = "1s", optimization = optimization)
pyblp$options$verbose <- FALSE
print(estimation_diagnostics(optimal_results))

# Estimates remain similar, while unobserved price heterogeneity is somewhat
# larger and more precisely estimated. Similar point estimates alone do not
# prove that the original instruments were close to efficient.

## ---- S5. Add an instrument and update the GMM weighting matrix --------------

product_data <- product_data %>%
  left_join(select(demographic_variation, market_ids, log_income_std),
            by = "market_ids") %>%
  mutate(demand_instruments4 = log_income_std * demand_instruments3)
overidentified_problem <- pyblp$Problem(
  product_formulations, product_data, agent_formulation, agent_data)
print(overidentified_problem)

pyblp$options$verbose <- TRUE
overidentified_results <- overidentified_problem$solve(
  sigma = sigma_start, pi = pi_start,
  method = "1s", optimization = optimization)
pyblp$options$verbose <- FALSE
print(estimation_diagnostics(overidentified_results))

# With more moments than parameters, the minimized objective is positive.
# A small gradient and positive Hessian support a local minimum; multiple
# starts would help assess whether there are better solutions elsewhere.

pyblp$options$verbose <- TRUE
overidentified_second_results <- overidentified_problem$solve(
  sigma = sigma_start, pi = pi_start,
  method = "1s", optimization = optimization,
  W = overidentified_results$updated_W)
pyblp$options$verbose <- FALSE
print(estimation_diagnostics(overidentified_second_results))

coefficient_table <- function(results) {
  data.frame(
    Parameter = c("Sigma on Price", "Pi on Mushy", "Pi on Price", "Alpha"),
    Estimate = c(py_to_r(results$sigma)[2, 2],
                 py_to_r(results$pi)[, 1], as.numeric(py_to_r(results$beta))),
    SE = c(py_to_r(results$sigma_se)[2, 2],
           py_to_r(results$pi_se)[, 1], as.numeric(py_to_r(results$beta_se))))
}
weighting_comparison <- merge(
  coefficient_table(overidentified_results),
  coefficient_table(overidentified_second_results),
  by = "Parameter", suffixes = c("_Step1", "_Step2"), sort = FALSE)
print(weighting_comparison)

# Estimates are similar; standard errors can improve with the updated W.

## ---- S6. Incorporate supply-side restrictions -------------------------------

product_data$firm_ids <- substr(product_data$product_ids, 1, 2)
supply_product_formulations <- tuple(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  pyblp$Formulation("0 + mushy + prices"),
  pyblp$Formulation("1 + mushy"))
supply_problem <- pyblp$Problem(
  supply_product_formulations, product_data, agent_formulation, agent_data)
print(supply_problem)

# Marginal cost is a constant + gamma_mushy * mushy + omega. The constant
# and mushy are automatically included as exogenous supply instruments.
# Alpha must now be optimized rather than concentrated out, so give beta
# a starting value. Gamma is still concentrated out by PyBLP.

pyblp$options$verbose <- TRUE
supply_results <- supply_problem$solve(
  sigma = sigma_start, pi = pi_start, beta = 10,
  method = "1s", optimization = optimization)
pyblp$options$verbose <- FALSE
print(estimation_diagnostics(supply_results))
print(data.frame(
  row.names = py_to_r(supply_results$gamma_labels),
  Estimate = as.numeric(py_to_r(supply_results$gamma)),
  SE = as.numeric(py_to_r(supply_results$gamma_se))))
  
# Mushy cereal has a lower estimated marginal cost. Demand estimates remain
# essentially the same as the overidentified first-step demand-only fit:
# these supply moments do not add identifying information for demand here.
