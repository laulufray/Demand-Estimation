# ==============================================================================
# Exercise 3 -- R code of Solutions.ipynb (Python + pyblp)
#
# One-time setup in the R console (skip if already done for Exercises 1-2):
#   install.packages(c("dplyr", "reticulate"))
#   library(reticulate)
#   python <- Sys.which(c("python3", "python"))
#   python <- python[nzchar(python)][1]
#   virtualenv_create("r-pyblp", python = python)
#   virtualenv_install("r-pyblp", c("pyblp", "pandas==2.2.3", "numpy", "statsmodels"))
#
# Run sections interactively, or from the repository root with:
#   source("Exercises/Exercise-3/Solutions.R")
# All three supplemental questions are included and take longer to run
# (the nesting parameter in S3 alone takes several minutes).
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

# Prefer the repository's data, whether working from the root or Exercise-3.
# Fall back to the same URLs as the notebook for other working directories.
read_exercise_data <- function(filename) {
  paths <- file.path(c("Exercises/Data", "../Data", "Data"), filename)
  available <- paths[file.exists(paths)]
  path <- if (length(available)) available[1] else paste0(
    "https://github.com/Mixtape-Sessions/Demand-Estimation/raw/main/Exercises/Data/",
    filename)
  read.csv(path, stringsAsFactors = FALSE)
}

# Shares in the counterfactual market after the F1B04 price cut
compute_counterfactual_shares <- function(results) {
  as.numeric(py_to_r(results$compute_shares(
    market_id = counterfactual_market, prices = counterfactual_data$new_prices)))
}
percent_change <- function(new, old) 100 * (new - old) / old

# Check the objective, first-order conditions, and second-order conditions.
estimation_diagnostics <- function(results) {
  data.frame(
    Converged = py_to_r(results$converged),
    Objective = as.numeric(py_to_r(results$objective)),
    Gradient_Norm = as.numeric(py_to_r(results$projected_gradient_norm)),
    Min_Hessian_Eigenvalue = min(py_to_r(results$reduced_hessian_eigenvalues)))
}

## ---- 0. Relevant code from Exercises 1 and 2 --------------------------------

# Exercise 1
product_data <- read_exercise_data("products.csv") %>%
  mutate(market_size = city_population * 90,
         shares = servings_sold / market_size) %>%
  rename(market_ids = market, product_ids = product,
         prices = price_per_serving)

first_stage <- lm(
  prices ~ 0 + price_instrument + factor(market_ids) + factor(product_ids),
  data = product_data)
product_data <- product_data %>%
  rename(demand_instruments0 = price_instrument)
iv_problem <- pyblp$Problem(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  product_data)
iv_results <- iv_problem$solve(method = "1s")

counterfactual_market <- "C01Q2"
counterfactual_data <- product_data %>%
  filter(market_ids == counterfactual_market) %>%
  select(product_ids, mushy, prices, shares)
counterfactual_data$new_prices <- counterfactual_data$prices
counterfactual_data$new_prices[counterfactual_data$product_ids == "F1B04"] <-
  counterfactual_data$new_prices[counterfactual_data$product_ids == "F1B04"] / 2
counterfactual_data$new_shares <- compute_counterfactual_shares(iv_results)
counterfactual_data$iv_change <- percent_change(
  counterfactual_data$new_shares, counterfactual_data$shares)

# Exercise 2.1
demographic_data <- read_exercise_data("demographics.csv") %>%
  rename(market_ids = market) %>%
  mutate(log_income = log(quarterly_income))
demographic_variation <- demographic_data %>%
  group_by(market_ids) %>%
  summarise(log_income_mean = mean(log_income),
            log_income_std = sd(log_income), .groups = "drop")

# Exercise 2.2: draw with replacement within each market using pandas'
# random_state = 0, so agents match the notebook's.
agent_data <- py_to_r(r_to_py(as.data.frame(
  select(demographic_data, market_ids, log_income)))$groupby(
    "market_ids", as_index = FALSE)$sample(
      n = 1000L, replace = TRUE, random_state = 0L)$reset_index(drop = TRUE))
agent_data[, c("nodes0", "nodes1", "nodes2")] <- py_to_r(
  np$random$default_rng(seed = 0L)$normal(
    size = tuple(as.integer(nrow(agent_data)), 3L)))
agent_data$weights <- 1 / 1000

product_data <- product_data %>%
  left_join(select(demographic_variation, market_ids, log_income_mean),
            by = "market_ids") %>%
  mutate(demand_instruments1 = log_income_mean * mushy)
agent_formulation <- pyblp$Formulation("0 + log_income")
mushy_problem <- pyblp$Problem(
  tuple(pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
        pyblp$Formulation("0 + mushy")),
  product_data, agent_formulation, agent_data)
optimization <- pyblp$Optimization(
  "trust-constr", list(gtol = 1e-8, xtol = 1e-8))
mushy_results <- mushy_problem$solve(
  sigma = 0, pi = 1, method = "1s", optimization = optimization)

# Exercise 2.4
counterfactual_data$new_shares <- compute_counterfactual_shares(mushy_results)
counterfactual_data$mushy_change <- percent_change(
  counterfactual_data$new_shares, counterfactual_data$shares)

# Exercise 2.5
product_data$predicted_prices <- as.numeric(fitted(first_stage))
compute_differentiation <- function(x) rowSums(outer(x, x, "-")^2)
product_data <- product_data %>%
  mutate(demand_instruments2 = log_income_mean * predicted_prices) %>%
  group_by(market_ids) %>%
  mutate(demand_instruments3 = compute_differentiation(predicted_prices)) %>%
  ungroup()
rc_problem <- pyblp$Problem(
  tuple(pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
        pyblp$Formulation("0 + mushy + prices")),
  product_data, agent_formulation, agent_data)
rc_results <- rc_problem$solve(
  sigma = diag(c(0, 1)), pi = matrix(c(0.2, 1), ncol = 1),
  method = "1s", optimization = optimization)

# Exercise 2.6
counterfactual_data$new_shares <- compute_counterfactual_shares(rc_results)
counterfactual_data$rc_change <- percent_change(
  counterfactual_data$new_shares, counterfactual_data$shares)

## ---- 1. Use the income statistic to match a parameter on log income ---------

# First, add a constant to the X2 formulation.
product_formulations <- tuple(
  pyblp$Formulation("0 + prices", absorb = "C(market_ids) + C(product_ids)"),
  pyblp$Formulation("1 + mushy + prices"))
micro_problem <- pyblp$Problem(
  product_formulations, product_data, agent_formulation, agent_data)
print(micro_problem)

# PyBLP calls each micro function with (t, p, a): a market ID and that
# market's products and agents. convert = FALSE passes them as Python
# objects (e.g. p$X2, a$demographics). We return R arrays with dimensions
# I x J, or I x J x (1 + J) when second choices k are involved (k = 0 is the
# outside option); reticulate converts these to NumPy arrays.
micro_function <- function(f) r_to_py(f, convert = FALSE)

# Repeat a J x (1 + J) matrix over the I agents in the first dimension.
expand_agents <- function(values, a) {
  I <- py_to_r(a$size)
  array(rep(values, each = I), c(I, dim(values)))
}

# Next, define the micro dataset.
survey_markets <- c("C01Q1", "C01Q2")
compute_income_weights <- micro_function(function(t, p, a) {
  matrix(1, py_to_r(a$size), py_to_r(p$size))
})
income_dataset <- pyblp$MicroDataset(
  "Income Survey", 100L, compute_income_weights, market_ids = survey_markets)
print(income_dataset)

# On it, define the micro part.
compute_income_values <- micro_function(function(t, p, a) {
  log_income <- py_to_r(a$demographics)[, 1]
  matrix(log_income, length(log_income), py_to_r(p$size))
})
income_part <- pyblp$MicroPart(
  "E[log_income_i | j > 0]", income_dataset, compute_income_values)
print(income_part)

# Using this, define the micro moment.
income_moment <- pyblp$MicroMoment("E[log_income_i | j > 0]", 7.9, income_part)
print(income_moment)

# Rows of Sigma and Pi follow X2 = (1, mushy, prices). The micro moment
# pins down the new parameter on 1 x log_income alone.
pyblp$options$verbose <- TRUE
micro_results <- micro_problem$solve(
  sigma = diag(c(0, 0, 6)),
  pi = matrix(c(1, 0.1, -6), ncol = 1),
  method = "1s", optimization = optimization,
  micro_moments = list(income_moment))
pyblp$options$verbose <- FALSE

# The new estimate (about -0.33, SE 0.55) is not significantly different
# from zero, suggesting our original assumption that it was zero was not too
# bad. This is not at all guaranteed: we may just have been lucky (or the
# instructor chose this imagined statistic to make this happen).

## ---- 2. Use the diversion statistics to estimate unobserved preference -----
## ----    heterogeneity for a constant and mushy -------------------------------

# First, define the new micro dataset.
compute_diversion_weights <- micro_function(function(t, p, a) {
  J <- py_to_r(p$size)
  expand_agents(matrix(1, J, 1 + J), a)
})
diversion_dataset <- pyblp$MicroDataset(
  "Diversion Survey", 200L, compute_diversion_weights,
  market_ids = survey_markets)
print(diversion_dataset)

# The first moment matches outside diversion.
compute_outside_values <- micro_function(function(t, p, a) {
  J <- py_to_r(p$size)
  expand_agents(cbind(1, matrix(0, J, J)), a)
})
outside_part <- pyblp$MicroPart(
  "P(k = 0 | j > 0)", diversion_dataset, compute_outside_values)
outside_moment <- pyblp$MicroMoment("P(k = 0 | j > 0)", 0.28, outside_part)
print(outside_moment)

# The second moment matches mushy diversion. Mushy is the second column
# of X2 in R (index 1 in Python).
compute_mushy_values <- micro_function(function(t, p, a) {
  mushy <- py_to_r(p$X2)[, 2]
  expand_agents(outer(mushy, c(0, mushy)), a)
})
mushy_part <- pyblp$MicroPart(
  "P(mushy_j and mushy_k | j > 0)", diversion_dataset, compute_mushy_values)
mushy_moment <- pyblp$MicroMoment(
  "P(mushy_j and mushy_k | j > 0)", 0.31, mushy_part)
print(mushy_moment)

# Re-optimize with the new micro moments, choosing some initial values for
# the new parameters.
pyblp$options$verbose <- TRUE
micro_results <- micro_problem$solve(
  sigma = diag(c(1, 1, 6)),
  pi = matrix(c(-0.3, 0.1, -6), ncol = 1),
  method = "1s", optimization = optimization,
  micro_moments = list(income_moment, outside_moment, mushy_moment))
pyblp$options$verbose <- FALSE
print(estimation_diagnostics(micro_results))

# All the standard optimization checks look fine. The new estimates suggest
# a good amount of unobserved preference heterogeneity for mushy (Sigma about
# 4.2), and some for the constant characteristic, i.e. the outside good
# (about 1.7).

## ---- 3. Evaluate changes to the price cut counterfactual --------------------

counterfactual_data$new_shares <- compute_counterfactual_shares(micro_results)
counterfactual_data$micro_change <- percent_change(
  counterfactual_data$new_shares, counterfactual_data$shares)
print(counterfactual_data)

# Substitution and cannibalization now look much more reasonable. There is
# much more substitution within mushy cereals: if the price of a mushy cereal
# drops, we expect mainly consumers of similar cereals to switch to it.

# ==============================================================================
# Supplemental Questions
# ==============================================================================

## ---- S1. See how your market size assumption affects results ----------------

# Drop the Sigma parameter on the constant, assuming no unobserved preference
# heterogeneity for the outside option, and drop its micro moment.
pyblp$options$verbose <- TRUE
restricted_results <- micro_problem$solve(
  sigma = diag(c(0, 4, 6)),
  pi = matrix(c(-0.3, 0.1, -6), ncol = 1),
  method = "1s", optimization = optimization,
  micro_moments = list(income_moment, mushy_moment))
pyblp$options$verbose <- FALSE

# Initialize a new problem with a market size twice as large as before.
alt_product_data <- product_data %>%
  mutate(market_size = market_size * 2,
         shares = servings_sold / market_size)
alt_problem <- pyblp$Problem(
  product_formulations, alt_product_data, agent_formulation, agent_data)
print(alt_problem)

# First, the same restricted model without outside-option heterogeneity.
pyblp$options$verbose <- TRUE
alt_restricted_results <- alt_problem$solve(
  sigma = diag(c(0, 4, 6)),
  pi = matrix(c(-0.3, 0.1, -6), ncol = 1),
  method = "1s", optimization = optimization,
  micro_moments = list(income_moment, mushy_moment))
pyblp$options$verbose <- FALSE

# Then add back this heterogeneity and the outside diversion ratio.
pyblp$options$verbose <- TRUE
alt_unrestricted_results <- alt_problem$solve(
  sigma = diag(c(2, 4, 6)),
  pi = matrix(c(-0.3, 0.1, -6), ncol = 1),
  method = "1s", optimization = optimization,
  micro_moments = list(income_moment, outside_moment, mushy_moment))
pyblp$options$verbose <- FALSE

# The notebook's text says to double all inside prices, but its code reuses
# the F1B04 price cut; we follow the code. The last row is the outside good.
alt_counterfactual_data <- counterfactual_data %>%
  mutate(
    alt_shares = alt_product_data$shares[
      alt_product_data$market_ids == counterfactual_market],
    restricted_shares = compute_counterfactual_shares(restricted_results),
    alt_restricted_shares = compute_counterfactual_shares(alt_restricted_results),
    alt_unrestricted_shares = compute_counterfactual_shares(alt_unrestricted_results))
alt_counterfactual_data <- bind_rows(
  alt_counterfactual_data,
  summarise(alt_counterfactual_data, across(
    c(shares, alt_shares, restricted_shares, alt_restricted_shares,
      alt_unrestricted_shares),
    ~ 1 - sum(.x)))) %>%
  mutate(restricted_change = percent_change(restricted_shares, shares),
         alt_restricted_change = percent_change(alt_restricted_shares, alt_shares),
         alt_unrestricted_change = percent_change(alt_unrestricted_shares, alt_shares))
print(select(alt_counterfactual_data, product_ids, mushy, prices, new_prices,
             restricted_change, alt_restricted_change, alt_unrestricted_change))

# Doubling the market size effectively increases the quality of the outside
# option, so there is less substitution away from it when an inside good
# becomes cheaper. Matching the outside diversion ratio pins down a different
# outside diversion estimate, with even less substitution away from the
# outside option. This could combine at least two things: with the extra
# parameter demand is slightly less elastic, so there is less substitution in
# general; and the "right" market size may be a bit larger than we assumed.

## ---- S2. Simulate some micro data and use it to match optimal micro moments -

# Simulate fake micro data, to show how we'd use a full micro dataset.
micro_data <- as.data.frame(py_to_r(pyblp$data_to_dict(
  micro_results$simulate_micro_data(income_dataset, seed = 0L))))
print(head(micro_data))

# micro_ids run from 0 to the number of observations minus one; market_ids
# are the observations' markets; agent_indices and choice_indices are
# zero-based row indices of the drawn agent types and choices in that
# market's agent and product data.
#
# With unobserved preference heterogeneity, we wouldn't observe the full
# type i in agent_indices, just income. Merge in log income and drop them.
agent_data <- agent_data %>%
  group_by(market_ids) %>%
  mutate(agent_indices = row_number() - 1) %>%
  ungroup()
micro_data <- micro_data %>%
  inner_join(select(agent_data, market_ids, agent_indices, log_income),
             by = c("market_ids", "agent_indices")) %>%
  select(-agent_indices)
print(head(micro_data))

# To compute scores, PyBLP needs to integrate over each observation's
# unobserved preference heterogeneity. As in agent_data, draw nodes and
# weights for each observation.
micro_data <- micro_data[rep(seq_len(nrow(micro_data)), times = 1000), ]
micro_data[, c("nodes0", "nodes1", "nodes2")] <- py_to_r(
  np$random$default_rng(0L)$normal(
    size = tuple(as.integer(nrow(micro_data)), 3L)))
micro_data$weights <- 1 / 1000
print(head(arrange(micro_data, micro_ids)))

# Compute the score of each observation, one vector per nonlinear parameter.
theta_labels <- py_to_r(micro_results$theta_labels)
micro_scores <- py_to_r(micro_results$compute_micro_scores(
  income_dataset, micro_data))
print(length(micro_scores))
print(theta_labels)
print(length(micro_scores[[1]]))

# Optimal micro moments also need scores for each market-type-choice
# combination (t, i, j) covered by the dataset. PyBLP replicates each consumer
# type with Monte Carlo draws, as we did above. For each parameter, this gives
# a list mapping market IDs to I x J matrices.
agent_scores <- py_to_r(micro_results$compute_agent_scores(
  income_dataset,
  integration = pyblp$Integration("monte_carlo", 1000L, list(seed = 0L))))
print(names(agent_scores[[1]]))
print(dim(agent_scores[[1]][["C01Q1"]]))

# Form the optimal micro moments. A Python lambda in a for loop needs an
# extra default argument to fix agent_scores_m; lapply gives each function
# its own m, so no such trick is needed here.
optimal_micro_moments <- lapply(seq_along(theta_labels), function(m) {
  name <- paste("Score for", theta_labels[m])
  pyblp$MicroMoment(
    name = name,
    value = mean(micro_scores[[m]]),
    parts = pyblp$MicroPart(
      name = name, dataset = income_dataset,
      compute_values = micro_function(
        function(t, p, a) agent_scores[[m]][[py_to_r(t)]])))
})
print(optimal_micro_moments)

# Using all six moments would overidentify the model, and we don't expect the
# income data to credibly identify Sigma. To keep estimates comparable, just
# replace the old sub-optimal income moment with its optimal one.
pyblp$options$verbose <- TRUE
optimal_results <- micro_problem$solve(
  sigma = diag(c(2, 4, 6)),
  pi = matrix(c(-0.3, 0.1, -6), ncol = 1),
  method = "1s", optimization = optimization,
  micro_moments = list(
    optimal_micro_moments[[which(theta_labels == "1 x log_income")]],
    outside_moment, mushy_moment))
pyblp$options$verbose <- FALSE

# Results are fairly similar. This is expected, since we simulated the micro
# data at our old estimates. The small micro dataset likely accounts for
# many of the differences.

## ---- S3. Use a within-firm diversion ratio to estimate a nesting parameter --

# Re-create the problem with nests equal to firm IDs: there are now H = 5.
product_data$nesting_ids <- substr(product_data$product_ids, 1, 2)
rcnl_problem <- pyblp$Problem(
  product_formulations, product_data, agent_formulation, agent_data)
print(rcnl_problem)

# Match the share of within-firm diversion on the diversion survey.
compute_firm_values <- micro_function(function(t, p, a) {
  firms <- as.character(py_to_r(p$nesting_ids))
  expand_agents(1 * outer(firms, c("", firms), "=="), a)
})
firm_part <- pyblp$MicroPart(
  "P(firm_j = firm_k | j > 0)", diversion_dataset, compute_firm_values)
firm_moment <- pyblp$MicroMoment("P(firm_j = firm_k | j > 0)", 0.35, firm_part)
print(firm_moment)

pyblp$options$verbose <- TRUE
rcnl_results <- rcnl_problem$solve(
  sigma = diag(c(2, 4, 6)),
  pi = matrix(c(-0.3, 0.1, -6), ncol = 1),
  rho = 0.1,
  method = "1s", optimization = optimization,
  micro_moments = list(income_moment, outside_moment, mushy_moment, firm_moment))
pyblp$options$verbose <- FALSE

# The nesting parameter is fairly low but nonzero (about 0.15), suggesting
# some but not a lot of unobserved firm-specific preferences. Sigma estimates
# are a bit lower, presumably because they previously picked up some
# unmodeled within-firm preferences.

counterfactual_data$new_shares <- compute_counterfactual_shares(rcnl_results)
counterfactual_data$rcnl_change <- percent_change(
  counterfactual_data$new_shares, counterfactual_data$shares)
print(counterfactual_data)

# With nontrivial firm-specific preferences, there is more within-firm
# substitution than before. Cannibalization in particular is larger relative
# to substitution from the firm's competitors, as we might expect.
