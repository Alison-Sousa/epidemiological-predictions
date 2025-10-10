# 1) LOAD PACKAGES
library(sf)
library(geobr)
library(dplyr)
library(readr)
library(stringr)
library(ggplot2)

# 2) READ AND PREPARE DATA
# Reading cluster data
clusters <- read_csv("C:/Users/PC GAMER/Downloads/dataset/results/clu.csv") %>%
  rename(cluster = clu)
clusters$cod6 <- as.character(clusters$cod6)

# Reading Brazil municipality map data
muni_map <- read_municipality(year = 2022, simplified = TRUE) %>%
  mutate(cod6 = str_sub(as.character(code_muni), 1, 6))

# Joining map data with clusters
map_clusters <- left_join(muni_map, clusters, by = "cod6")

# 3) DEFINE STATES AND CLASSIFY RISK
# Southeast states information
state_info <- tibble(
  code_state = c(31, 32, 33, 35),
  state = c("MG", "ES", "RJ", "SP")
)

# Function to classify risk based on probability
classify_risk <- function(prob) {
  case_when(
    prob >= 0.8 ~ "Very High",
    prob >= 0.6 ~ "High",
    prob >= 0.4 ~ "Medium",
    prob >= 0.2 ~ "Low",
    prob >= 0.0 ~ "Very Low"
  )
}

# Risk levels for ordering
risk_levels <- c("Very High", "High", "Medium", "Low", "Very Low")

# Filter for Southeast and apply risk classification
map_sudeste <- map_clusters %>%
  filter(code_state %in% state_info$code_state) %>%
  left_join(state_info, by = "code_state") %>%
  mutate(
    risk_cat = classify_risk(prob),
    risk_cat = factor(risk_cat, levels = risk_levels)
  )

# 4) DEFINE COLOR PALETTE
colors <- c(
  "Very High" = "#d73027",
  "High"      = "#fc8d59",
  "Medium"    = "#ffffbf",
  "Low"       = "#91bfdb",
  "Very Low"  = "#4575b4"
)

# 5) FUNCTION TO CALCULATE CORRECT ZOOM (BOUNDING BOX)
mainland_bbox <- function(sfobj, pad = 0.06) {
  geom <- st_geometry(sfobj)
  u <- st_union(geom)
  polys <- st_cast(u, "POLYGON", warn = FALSE)
  areas <- st_area(polys)
  main_poly <- polys[which.max(areas)]
  bb <- st_bbox(main_poly)
  dx <- as.numeric(bb$xmax - bb$xmin)
  dy <- as.numeric(bb$ymax - bb$ymin)
  xlim <- c(bb$xmin - pad * dx, bb$xmax + pad * dx)
  ylim <- c(bb$ymin - pad * dy, bb$ymax + dy * pad)
  list(xlim = xlim, ylim = ylim)
}

# 6) SINGLE PLOTTING FUNCTION (SIMPLIFIED)
# This function no longer has an option to show a legend
plot_state_individual <- function(code, abbr) {
  
  dados <- map_sudeste %>% filter(code_state == code)
  bb <- mainland_bbox(dados, pad = 0.08)
  
  # Create the base plot
  p <- ggplot() +
    geom_sf(data = dados, aes(fill = risk_cat), color = NA) +
    scale_fill_manual(
      values = colors, 
      drop = FALSE, 
      name = NULL, 
      na.value = "#777777"
    ) +
    coord_sf(xlim = bb$xlim, ylim = bb$ylim, expand = FALSE) +
    ggtitle(abbr) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 28, face = "bold", margin = margin(b = 10)),
      plot.margin = unit(c(0, 0, 0, 0), "cm"),
      legend.position = "none" # Remove legend from all plots
    )
  
  return(p)
}

# 7) LOOP TO CREATE AND SAVE INDIVIDUAL MAPS
output_dir <- "C:/Users/PC GAMER/Downloads/dataset/results/"

# Make sure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Loop that generates the maps
for (i in 1:nrow(state_info)) {
  code <- state_info$code_state[i]
  abbr <- state_info$state[i]
  
  # The function call was simplified
  final_map <- plot_state_individual(code, abbr)
  
  # Save the map to a PDF file
  ggsave(
    filename = paste0(output_dir, "risk_map_", abbr, ".pdf"), # Changed filename
    plot = final_map,
    width = 10, 
    height = 10,
    device = cairo_pdf
  )
  
  message(paste("Map for", abbr, "saved successfully!")) # Changed message
}

