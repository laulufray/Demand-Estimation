# ==============================================================================
# Exercise 1 -- R code of Solutions.ipynb (Python + pyblp)
#
# Call the real Python pyblp package from R via `reticulate`, so results are
# numerically identical to the notebook.
#
# One-time setup:
#   library(reticulate)
#   virtualenv_create("r-pyblp", python = "/opt/homebrew/bin/python3")
#   virtualenv_install("r-pyblp", c("pyblp", "pandas==2.2.3", "numpy", "statsmodels"))
#
# ==============================================================================

library(dplyr)
library(reticulate)
library(sandwich)
library(lmtest)

use_virtualenv("r-pyblp", required = TRUE)
pyblp <- import("pyblp", convert = FALSE)
pyblp$options$digits <- 3L
pyblp$options$verbose <- FALSE

set.seed(0)

## ---- 1. Describe the data --------------------------------------------------

product_data <- read.csv("https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/products.csv")
product_data[sample(nrow(product_data), 5), ]

## ---- 2. Compute market shares ----------------------------------------------

product_data <- product_data %>%
  mutate(market_size = city_population * 90,
         market_share = servings_sold / market_size) %>%
  group_by(market) %>%
  mutate(outside_share = 1 - sum(market_share)) %>%
  ungroup()

summary(product_data[, c("market_share", "outside_share")])

## ---- 3. Estimate the pure logit model with OLS ------------------------------
## Native R: matches statsmodels' smf.ols(...).fit(cov_type='HC0') exactly.

product_data <- product_data %>%
  mutate(logit_delta = log(market_share / outside_share))

ols_fit <- lm(logit_delta ~ 1 + mushy + price_per_serving, data = product_data)
ols_hc0 <- coeftest(ols_fit, vcov = vcovHC(ols_fit, type = "HC0"))
ols_hc0

# The coefficient on price is negative, so demand slopes down. To interpret
# the mushy coefficient, divide it by the price coefficient: this is the
# willingness to pay of consumers for a cereal being "mushy."
cat(sprintf("\nWTP for 'mushy' = %.3f\n", coef(ols_fit)["mushy"] / -coef(ols_fit)["price_per_serving"]))

## ---- 4. Run the same regression with pyblp ----------------------------------
## From here on we rename columns to pyblp's expected names and keep using this 
## renamed `product_data` for the rest of the script.

product_data <- product_data %>%
  rename(market_ids = market, product_ids = product,
         shares = market_share, prices = price_per_serving)
product_data$demand_instruments0 <- product_data$prices

ols_problem <- pyblp$Problem(pyblp$Formulation("1 + mushy + prices"), product_data)
print(ols_problem)

# Double-check that pyblp's instrument matrix is a constant, mushy, and
# prices. The ordering is different, but it's the same.
ZD <- as.data.frame(py_to_r(ols_problem$products$ZD))
ZD[sample(nrow(ZD), 5), ]

ols_results <- ols_problem$solve(method = "1s")
print(ols_results)

# Compare estimates and SEs side by side.
beta_labels <- py_to_r(ols_results$beta_labels)
data.frame(
  row.names = beta_labels,
  Estimate_Statsmodels = coef(ols_fit),
  Estimate_PyBLP = as.numeric(py_to_r(ols_results$beta)),
  SE_Statsmodels = ols_hc0[, "Std. Error"],
  SE_PyBLP = as.numeric(py_to_r(ols_results$beta_se))
)
# We get the same estimates and the same standard errors.

## ---- 5. Add market and product fixed effects ---------------------------------
## Fixed effects are absorbed via iterative de-meaning, dropping the constant
## and mushy dummy since they're collinear with the fixed effects.

fe_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  product_data)
print(fe_problem)

fe_results <- fe_problem$solve(method = "1s")
print(fe_results)

# We get a more negative price coefficient, suggesting the OLS coefficient
# was biased upward: price was positively correlated with product/market-
# specific unobserved quality.

## ---- 6. Add an instrument for price ------------------------------------------

# First stage: is the price instrument relevant? (native R, matches statsmodels)
first_stage <- lm(prices ~ 0 + price_instrument + factor(market_ids) + factor(product_ids),
                   data = product_data)
first_stage_hc0 <- coeftest(first_stage, vcov = vcovHC(first_stage, type = "HC0"))
first_stage_hc0[order(rownames(first_stage_hc0), decreasing = TRUE), ]

# It's strongly relevant even after adjusting for fixed effects. Now use it
# to instrument for price via pyblp.
product_data <- product_data %>%
  select(-demand_instruments0) %>%
  rename(demand_instruments0 = price_instrument)

iv_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  product_data)
print(iv_problem)

iv_results <- iv_problem$solve(method = "1s")
print(iv_results)

data.frame(
  row.names = py_to_r(fe_results$beta_labels),
  Estimate_OLS = as.numeric(py_to_r(ols_results$beta))[length(as.numeric(py_to_r(ols_results$beta)))],
  Estimate_FE = as.numeric(py_to_r(fe_results$beta)),
  Estimate_IV = as.numeric(py_to_r(iv_results$beta)),
  SE_OLS = as.numeric(py_to_r(ols_results$beta_se))[length(as.numeric(py_to_r(ols_results$beta_se)))],
  SE_FE = as.numeric(py_to_r(fe_results$beta_se)),
  SE_IV = as.numeric(py_to_r(iv_results$beta_se))
)
# Our estimate gets even more negative with an IV, suggesting the within
# product-and-market component of unobserved quality was still positively
# correlated with price.

## ---- 7. Cut a price in half and see what happens ------------------------------

counterfactual_market <- "C01Q2"
counterfactual_data <- product_data %>%
  filter(market_ids == counterfactual_market) %>%
  select(product_ids, mushy, prices, shares)
counterfactual_data

counterfactual_data$new_prices <- counterfactual_data$prices
counterfactual_data$new_prices[counterfactual_data$product_ids == "F1B04"] <-
  counterfactual_data$new_prices[counterfactual_data$product_ids == "F1B04"] / 2

new_shares <- iv_results$compute_shares(
  market_id = counterfactual_market,
  prices = counterfactual_data$new_prices)
counterfactual_data$new_shares <- as.numeric(py_to_r(new_shares))
counterfactual_data$iv_change <- 100 * (counterfactual_data$new_shares - counterfactual_data$shares) /
  counterfactual_data$shares
counterfactual_data
# The product whose price we halved gains a lot of share; every other
# product loses the same percent, which is an unrealistic substitution
# pattern: pure logit gives no extra substitution toward similar products.

## ---- 8. Compute demand elasticities --------------------------------------------

iv_elasticities <- py_to_r(iv_results$compute_elasticities(market_id = counterfactual_market))
rownames(iv_elasticities) <- colnames(iv_elasticities) <- counterfactual_data$product_ids
round(iv_elasticities, 3)
# Own-price elasticities suggest fairly elastic demand. Cross-price
# elasticities are small and, again, identical within each column:
# the same unrealistic IIA substitution pattern as above.

# ==============================================================================
# Supplemental Questions
# ==============================================================================

## ---- S1. Try different standard errors -----------------------------------------
## There are likely many market-varying unobserved characteristics (e.g.
## advertising), so it may matter to cluster by product.

product_data$clustering_ids <- product_data$product_ids
cluster_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  product_data)
cluster_results <- cluster_problem$solve(method = "1s", se_type = "clustered")

data.frame(
  row.names = py_to_r(fe_results$beta_labels),
  Estimate_Unclustered = as.numeric(py_to_r(iv_results$beta)),
  SE_Unclustered = as.numeric(py_to_r(iv_results$beta_se)),
  Estimate_Clustered = as.numeric(py_to_r(cluster_results$beta)),
  SE_Clustered = as.numeric(py_to_r(cluster_results$beta_se))
)
# Standard errors are somewhat larger, as expected: unaccounted-for error
# correlation typically biases standard errors downward.

## ---- S2. Compute confidence intervals for your counterfactual ------------------
## For speed, use 100 bootstrap draws (use more in practice).

bootstrap_results <- cluster_results$bootstrap(draws = 100L, seed = 0L)
print(bootstrap_results)

market_mask <- product_data$market_ids == counterfactual_market
bootstrap_shares_full <- py_to_r(bootstrap_results$bootstrapped_shares)  # (draws, N, 1)
bootstrap_shares <- bootstrap_shares_full[, market_mask, 1]              # (draws, J)
dim(bootstrap_shares)

bootstrap_new_prices <- matrix(counterfactual_data$new_prices, nrow = 100,
                                ncol = length(counterfactual_data$new_prices),
                                byrow = TRUE)
bootstrap_new_shares <- py_to_r(bootstrap_results$compute_shares(
  market_id = counterfactual_market,
  prices = r_to_py(bootstrap_new_prices)))
bootstrap_new_shares <- bootstrap_new_shares[, , 1]                      # (draws, J)

bootstrap_changes <- 100 * (bootstrap_new_shares - bootstrap_shares) / bootstrap_shares

counterfactual_data$iv_change_lb <- apply(bootstrap_changes, 2, quantile, probs = 0.025)
counterfactual_data$iv_change_ub <- apply(bootstrap_changes, 2, quantile, probs = 0.975)
counterfactual_data
# The confidence intervals are fairly tight: the price coefficient has a
# fairly low standard error, even after clustering.

## ---- S3. Impute marginal costs from pricing optimality --------------------------

product_data$firm_ids <- substr(product_data$product_ids, 1, 2)
firm_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  product_data)
firm_results <- firm_problem$solve(method = "1s", se_type = "clustered")

product_data$costs <- as.numeric(py_to_r(firm_results$compute_costs()))
product_data$profit_per_serving <- product_data$prices - product_data$costs
product_data$markups <- product_data$profit_per_serving / product_data$costs
summary(product_data[, c("prices", "costs", "profit_per_serving", "markups")])
# Marginal costs are positive and markups are on the order of 30%-60%,
# which look reasonable -- though our demand elasticities aren't very
# realistic yet, so these cost estimates shouldn't be trusted too much.

## ---- S4. Check your code by simulating data --------------------------------------
## Simulate new prices and shares under a less elastic price coefficient,
## alpha = -20, calibrated to our estimated fixed effects and unobserved
## quality, then see if we can recover it.

simulation <- pyblp$Simulation(
  product_formulations = pyblp$Formulation("0 + prices"),
  product_data = product_data,
  beta = -20,
  xi = iv_results$xi_fe + iv_results$xi)
print(simulation)

simulation_results <- simulation$replace_endogenous(costs = product_data$costs)
print(simulation_results)
# First order conditions (profit gradient norms) near zero and second order
# conditions (profit hessian eigenvalues) negative confirm the solved
# prices are profit-maximizing.

simulation_data <- product_data
simulation_data$shares <- as.numeric(py_to_r(simulation_results$product_data$shares))
simulation_data$prices <- as.numeric(py_to_r(simulation_results$product_data$prices))
simulation_data$demand_instruments0 <- simulation_data$costs

simulation_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  simulation_data)
simulation_problem_results <- simulation_problem$solve(method = "1s", se_type = "clustered")
print(simulation_problem_results)
# Our estimate should not be significantly different from the true
# alpha = -20, as hoped.
