library(sf)
library(geobr)
library(dplyr)
library(readr)
library(spdep)
library(ggplot2)
library(stringr)

# 1. Load and prepare data
# ---------------------------

# Load risk predictions and calculate mean per municipality
probs <- read_csv("C:/Users/PC GAMER/Downloads/dataset/data/R/prb.csv",
                  col_types = cols(
                    cod6 = col_character(),
                    municipio = col_character(),
                    year = col_integer(),
                    prob = col_double()
                  ))

probs_mean <- probs %>%
  group_by(cod6) %>%
  summarise(mean_risk = mean(prob, na.rm = TRUE))

# Load municipalities shapefile
map_muni <- read_municipality(code_muni = "all", year = 2022, simplified = TRUE) %>%
  mutate(cod6 = str_sub(as.character(code_muni), 1, 6))

# Join risk data to map and filter out missing values
map_data <- map_muni %>%
  left_join(probs_mean, by = "cod6") %>%
  filter(!is.na(mean_risk))

# Simplify geometry and remove empty features
map_simp <- st_simplify(map_data, dTolerance = 500, preserveTopology = TRUE)
map_simp <- map_simp[!st_is_empty(map_simp), ]


# 2. Spatial Autocorrelation Analysis (Moran's I)
# --------------------------------------------------

# Convert to sp object for spdep
sp_data <- sf::as_Spatial(map_simp)

# Create Queen contiguity spatial weights
nb <- poly2nb(sp_data, queen = TRUE)
lw <- nb2listw(nb, style = "W", zero.policy = TRUE)

# Global Moran's I test
set.seed(123)
moran_test <- moran.mc(map_simp$mean_risk, lw, nsim = 999, zero.policy = TRUE)
print(moran_test)

# Local Moran's I (LISA)
lisa_test <- localmoran(map_simp$mean_risk, lw, zero.policy = TRUE)

# Add LISA results to the sf object
map_simp$lisa_I <- lisa_test[, 1]
map_simp$lisa_p <- lisa_test[, 5]

# Center risk values for cluster classification
risk_centered <- map_simp$mean_risk - mean(map_simp$mean_risk, na.rm = TRUE)
lag_risk <- lag.listw(lw, map_simp$mean_risk, zero.policy = TRUE)
lag_risk_centered <- lag_risk - mean(lag_risk, na.rm = TRUE)

# Classify LISA clusters
map_simp$lisa_cluster <- "Not significant" # Default value
map_simp$lisa_cluster[risk_centered > 0 & lag_risk_centered > 0 & map_simp$lisa_p < 0.05] <- "High-High"
map_simp$lisa_cluster[risk_centered > 0 & lag_risk_centered < 0 & map_simp$lisa_p < 0.05] <- "High-Low"
map_simp$lisa_cluster[risk_centered < 0 & lag_risk_centered > 0 & map_simp$lisa_p < 0.05] <- "Low-High"
map_simp$lisa_cluster[risk_centered < 0 & lag_risk_centered < 0 & map_simp$lisa_p < 0.05] <- "Low-Low"

map_simp$lisa_cluster <- factor(
  map_simp$lisa_cluster,
  levels = c("High-High", "High-Low", "Low-High", "Low-Low", "Not significant")
)


# 3. Visualization and Export
# ------------------------------

# Define color palette for LISA map
lisa_palette <- c(
  "High-High"       = "#d73027",
  "High-Low"        = "#fdae61",
  "Low-High"        = "#abd9e9",
  "Low-Low"         = "#4575b4",
  "Not significant" = "#cccccc"
)

# Generate and save LISA map
lisa_map_plot <- ggplot(map_simp) +
  geom_sf(aes(fill = lisa_cluster), color = NA) +
  scale_fill_manual(values = lisa_palette, name = "LISA Cluster") +
  theme_void() +
  theme(
    legend.position = "left",
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 12),
    plot.margin = grid::unit(c(0, 0, 0, 0), "pt")
  )

pdf("C:/Users/PC GAMER/Downloads/dataset/results/lisa_map.pdf", width = 6, height = 5, useDingbats = FALSE)
print(lisa_map_plot)
dev.off()

# Export Global Moran's I summary
moran_summary <- data.frame(
  moran_I = as.numeric(moran_test$statistic),
  p_value = as.numeric(moran_test$p.value)
)
write_csv(moran_summary, "C:/Users/PC GAMER/Downloads/dataset/results/moran_summary.csv")

# Export detailed LISA results
lisa_results <- st_drop_geometry(map_simp)
write_csv(lisa_results, "C:/Users/PC GAMER/Downloads/dataset/results/lisa_results.csv")

# Export top 20 municipalities by mean risk
top20_risk <- lisa_results %>%
  arrange(desc(mean_risk)) %>%
  mutate(short_name = str_sub(name_muni, 1, 18)) %>%
  select(code_muni, short_name, abbrev_state, mean_risk, lisa_cluster) %>%
  head(20)

write_csv(top20_risk, "C:/Users/PC GAMER/Downloads/dataset/results/top20_risk.csv")

