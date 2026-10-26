# Independent check of the Figure of Merit computed in benchmark_scoring.py: the same maps
# scored with lulcc's FigureOfMerit (Moulds et al. 2015), which implements the three-map
# comparison of Pontius et al. (2011).
# lulcc needs too much memory for the whole 8.4-million-cell grid on a 16 GB machine, so both
# are compared on one 1,000 x 1,000-pixel window (rows and columns 1001-2000, central Guinea);
# benchmark_scoring.py scores the same window in notebook 00.
#
# How to run (about a minute): as benchmark_lulcc.R (RStudio: Source; Rgui.exe: File > Source R
# code...; terminal in landuse_experiments/: & "D:\envs\R\bin\Rscript.exe" scripts\benchmark_fom_check.R).
# Everything printed is
# also written to outputs/benchmark/output_lulcc/fom_check_log.txt.

if (dir.exists("D:/envs/R-library")) .libPaths(c("D:/envs/R-library", .libPaths()))
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) NULL)      # source()
if (is.null(here)) {                                                                       # Rscript
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  here <- if (length(f)) dirname(normalizePath(f)) else getwd()
}
setwd(dirname(here))                                  # landuse_experiments/ (this script is in scripts/)

suppressMessages(library(lulcc))
bench <- "outputs/benchmark"
sink(file.path(bench, "output_lulcc", "fom_check_log.txt"), split = TRUE)
cat("Figure of Merit with lulcc", as.character(packageVersion("lulcc")), "on", R.version.string,
    "\nRun:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n\n")
window <- function(r) crop(r, extent(r, 1001, 2000, 1001, 2000))
read_lc <- function(path) { r <- window(raster(path)); r[r == 0] <- NA; r }
start <- read_lc(file.path(bench, "inputs", "landcover_2024.tif"))
observed <- read_lc(file.path(bench, "reference", "landcover_2025_reference.tif"))
forecasts <- c("Deployed method" = file.path(bench, "output_ours", "deployed_2025.tif"),
               "Improved method" = file.path(bench, "output_ours", "improved_2025.tif"),
               "CLUE-S (lulcc)" = file.path(bench, "output_lulcc", "lulcc_clues_2025.tif"),
               "MOLUSCE LR" = file.path(bench, "output_molusce_2", "molusce_lr_2025_2.tif"))
for (name in names(forecasts)) {
  sim <- read_lc(forecasts[[name]])
  ok <- !is.na(start) & !is.na(observed) & !is.na(sim)   # the pixels benchmark_scoring.py scores
  fom <- FigureOfMerit(x = mask(start, ok, maskvalue = 0), x1 = mask(observed, ok, maskvalue = 0),
                       y1 = mask(sim, ok, maskvalue = 0), factors = 1, categories = 1:7,
                       labels = c("Water", "Trees", "Flooded", "Crops", "Built", "Bare", "Rangeland"))
  cat(sprintf("%-18s Figure of Merit (lulcc): %.2f%%\n", name, 100 * fom@overall[[1]]))
  rm(sim, ok, fom); invisible(gc())
}
sink()
