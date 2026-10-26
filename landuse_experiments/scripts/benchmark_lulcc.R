# CLUE-S with the R package lulcc (Moulds et al. 2015), on the benchmark inputs
# (outputs/benchmark/, written by benchmark_export.py). This run is the CLUE-S column of the
# tool comparison (notebook 00). The authors' original program and the Python version
# (clue_s.py) are checks.
#
# Same set-up as clue_s.py: one logistic regression per class on the 2024 map and the 2024
# predictors; conversion elasticity of each class = share of its pixels unchanged in
# 2023-2024; demand = the observed 2025 total of each class; every conversion allowed.
#
# How to run (about 7 minutes; close other heavy programs, it needs about 10 GB of memory):
#   * RStudio: open this file, click Source.
#   * R's own window (D:\envs\R\bin\x64\Rgui.exe): File > Source R code..., pick this file.
#   * A terminal in landuse_experiments/:  & "D:\envs\R\bin\Rscript.exe" scripts\benchmark_lulcc.R
# Needs R >= 4 and install.packages("lulcc").
#
# Everything printed is also written to outputs/benchmark/output_lulcc/run_log.txt, with the
# R and lulcc versions, the start and end times, and a checksum (MD5) of every output file.
# The run is seeded, so a rerun gives files with the same checksums.
#
# DEMAND (below): "observed", the observed 2025 total of each class (the comparison in notebook
# 00), or "projected", 2023-2024 projected one step by a Markov chain, as a user would have
# (notebook 10; output in output_lulcc/projected/). In a terminal it can be given instead:
#   & "D:\envs\R\bin\Rscript.exe" scripts\benchmark_lulcc.R projected

DEMAND <- "observed"
if ("projected" %in% commandArgs(trailingOnly = TRUE)) DEMAND <- "projected"
if (dir.exists("D:/envs/R-library")) .libPaths(c("D:/envs/R-library", .libPaths()))
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)      # source()
if (is.null(here)) {                                                                       # Rscript
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  here <- if (length(f)) dirname(normalizePath(f)) else getwd()
}
setwd(dirname(here))                                  # landuse_experiments/ (this script is in scripts/)

suppressMessages(library(lulcc))
set.seed(0)
bench <- "outputs/benchmark"
out <- if (DEMAND == "projected") file.path(bench, "output_lulcc", "projected") else file.path(bench, "output_lulcc")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
sink(file.path(out, "run_log.txt"), split = TRUE)
cat("CLUE-S with lulcc", as.character(packageVersion("lulcc")), "on", R.version.string, "\n")
cat("Script:", file.path(here, "benchmark_lulcc.R"), "\nDemand:", DEMAND, "\nStarted:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n\n")
labels <- c("Water", "Trees", "Flooded", "Crops", "Built", "Bare", "Rangeland")

read_lc <- function(path) {
  r <- raster(path)
  r[r == 0] <- NA                                       # 0 = outside Guinea
  r
}
lc23 <- read_lc(file.path(bench, "inputs", "landcover_2023.tif"))
lc24 <- read_lc(file.path(bench, "inputs", "landcover_2024.tif"))

obs <- ObsLulcRasterStack(x = stack(lc24), categories = 1:7, labels = labels, t = 0)
factor_files <- sort(list.files(file.path(bench, "inputs", "factors_2024"), pattern = "\\.tif$", full.names = TRUE))
factors <- stack(factor_files)
NAvalue(factors) <- -9999
names(factors) <- sprintf("ef_%03d", seq_along(factor_files))
ef <- ExpVarRasterList(x = factors)
cat("Predictors:", basename(factor_files), sep = "\n  ")

# Logistic regressions, fitted on 7.5% of the cells (about 300,000, as clue_s.py)
part <- partition(x = obs[[1]], size = 0.075, spatial = FALSE)
train <- getPredictiveModelInputData(obs = obs, ef = ef, cells = part[["train"]], t = 0)
rhs <- paste(names(factors), collapse = " + ")
forms <- lapply(labels, function(l) as.formula(paste(l, "~", rhs)))
models <- glmModels(formula = forms, family = binomial, data = train, obs = obs)

# Conversion elasticities and demand
v23 <- getValues(lc23); v24 <- getValues(lc24)
both <- !is.na(v23) & !is.na(v24)
elas <- sapply(1:7, function(k) sum(v23 == k & v24 == k & both) / max(sum(v23 == k & both), 1))
rm(train); invisible(gc())
legend <- read.csv(file.path(bench, "legend_and_demand.csv"))
now <- sapply(1:7, function(k) sum(v24 == k, na.rm = TRUE))
observed <- as.numeric(legend[[paste0("pixels_2025_", DEMAND)]])
target <- observed * sum(now) / sum(observed)
demand <- rbind(now, target)
cat("\nElasticities:", round(elas, 3), "\nDemand 2024:", now, "\nDemand 2025:", round(target), "\n")

# lulcc's CLUE-S allocation. Parameters: the package example's, with the convergence limits
# (in cells in lulcc) set from the original program's run: largest difference 5% of the
# smallest demand, average 1% of the mean demand (clue_s.py).
params <- lulcc:::.checkCluesParams(list(jitter.f = 0.0002, scale.f = 0.000001, max.iter = 5000,
                                         max.diff = ceiling(0.05 * min(target)),
                                         ave.diff = ceiling(0.01 * mean(target))))
# The steps of lulcc's allocate() for a CluesModel (lulcc 1.0.4), one time step, with the
# probabilities predicted in chunks: allocate() predicts all 4 million cells at once, which
# ran out of memory on a 16 GB machine. No decision rules, neighbourhood rules or mask are
# used, so those steps of allocate() change nothing and are left out.
started <- Sys.time()
cells <- which(!is.na(v24))
map0.vals <- v24[cells]
rm(v23, both); invisible(gc())
prob <- matrix(NA_real_, length(cells), 7)
for (start in seq(1, length(cells), by = 500000)) {
  rows <- start:min(start + 499999, length(cells))
  newdata <- as.data.frame(lapply(seq_len(nlayers(factors)), function(j) getValues(factors[[j]])[cells[rows]]))
  names(newdata) <- names(factors)
  prob[rows, ] <- predict(object = models, newdata = newdata)
}
tprob <- prob
for (j in 1:7) {
  ix <- map0.vals %in% j
  tprob[ix, j] <- tprob[ix, j] + elas[j]
}
auto <- lulcc:::.autoConvert(x = map0.vals, prob = tprob, categories = 1:7, mask = rep(1, length(map0.vals)))
stopifnot(length(auto$ix) == 0)
map1.vals <- do.call(lulcc:::.clues, c(list(tprob = tprob, map0.vals = map0.vals, demand = target,
                                            categories = 1:7), params))
cat("Allocation:", round(as.numeric(difftime(Sys.time(), started, units = "mins")), 1), "min\n")

sim <- lc24
sim[cells] <- map1.vals
alloc <- sapply(1:7, function(k) sum(map1.vals == k, na.rm = TRUE))
print(data.frame(class = labels, demand = round(target), allocated = alloc,
                 difference_pct = round(100 * (alloc - target) / target, 2)))
writeRaster(sim, file.path(out, "lulcc_clues_2025.tif"), datatype = "INT1U", NAflag = 0, overwrite = TRUE)
saveRDS(lapply(models@models, coef), file.path(out, "lulcc_regression_coefficients.rds"))
for (j in 1:7) {                                        # the regressions' probability maps
  p <- raster(lc24)
  p[cells] <- prob[, j]
  writeRaster(p, file.path(out, sprintf("lulcc_probability_%d.tif", j)), datatype = "FLT4S",
              NAflag = -1, overwrite = TRUE)
}
cat("Saved", file.path(out, "lulcc_clues_2025.tif"), "\n")

cat("\nFinished:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\nOutput files (MD5 checksum):\n")
files <- list.files(out, pattern = "\\.(tif|rds)$", full.names = TRUE)
print(data.frame(file = basename(files), md5 = unname(tools::md5sum(files))), row.names = FALSE)
cat("\n")
print(sessionInfo())
sink()
