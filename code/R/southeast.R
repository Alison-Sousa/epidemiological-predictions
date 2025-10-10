# Load required libraries
library(sf)
library(geobr)
library(dplyr)
library(readr)
library(stringr)
library(ggplot2)
library(cowplot)
library(purrr)

# 1) Read cluster data and correct column name
clusters <- read_csv("C:/Users/PC GAMER/Downloads/dataset/results/clu.csv") %>%
  rename(cluster = clu) # Rename 'clu' to 'cluster'
clusters$cod6 <- as.character(clusters$cod6)

# 2) Read Brazil municipality map data
muni_map <- read_municipality(year = 2022, simplified = TRUE) %>%
  mutate(cod6 = str_sub(as.character(code_muni), 1, 6))

# 3) Merge map with cluster data
map_clusters <- left_join(muni_map, clusters, by = "cod6")

# 4) Define Southeast states
state_info <- tibble(
  code_state = c(31, 32, 33, 35),
  state = c("MG", "ES", "RJ", "SP")
)

# 5) Function to classify risk (now returns NA for missing values)
classify_risk <- function(prob) {
  case_when(
    prob >= 0.8 ~ "Very High",
    prob >= 0.6 ~ "High",
    prob >= 0.4 ~ "Medium",
    prob >= 0.2 ~ "Low",
    prob >= 0.0 ~ "Very Low"
    # When 'prob' is NA, case_when automatically returns NA. No 'TRUE' line needed.
  )
}

# Define the desired order for the legend (without "No Data")
risk_levels <- c("Very High", "High", "Medium", "Low", "Very Low")

# Filter for Southeast, classify risk, and set legend order
map_sudeste <- map_clusters %>%
  filter(code_state %in% state_info$code_state) %>%
  left_join(state_info, by = "code_state") %>%
  mutate(
    risk_cat = classify_risk(prob),
    risk_cat = factor(risk_cat, levels = risk_levels)
  )

# 6) Define color palette (without "No Data")
colors <- c(
  "Very High" = "#d73027",
  "High"      = "#fc8d59",
  "Medium"    = "#ffffbf",
  "Low"       = "#91bfdb",
  "Very Low"  = "#4575b4"
)

# 7) Function to get bounding box of the largest landmass
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

# 8) Function to plot a single state map
plot_estado <- function(code, abbr) {
  dados <- map_sudeste %>% filter(code_state == code)
  bb <- mainland_bbox(dados, pad = 0.08)
  ggplot() +
    geom_sf(data = dados, aes(fill = risk_cat), color = NA) +
    # na.value now correctly handles all missing data
    scale_fill_manual(values = colors, drop = FALSE, name = NULL, na.value = "#777777") +
    coord_sf(xlim = bb$xlim, ylim = bb$ylim, expand = FALSE, datum = NA) +
    ggtitle(abbr) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 24, face = "bold"),
      legend.position = "none"
    )
}

# Create maps for each state
mg_map <- plot_estado(31, "MG")
es_map <- plot_estado(32, "ES")
rj_map <- plot_estado(33, "RJ")
sp_map <- plot_estado(35, "SP")

# 9) Create a single, shared legend
legend_plot <- ggplot() +
  geom_sf(data = map_sudeste, aes(fill = risk_cat), color = NA) +
  scale_fill_manual(values = colors, drop = FALSE, name = NULL, na.value = "#777777") +
  guides(fill = guide_legend(override.aes = list(color = NA))) + # Ensure legend patches have no borders
  theme_void() +
  theme(
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(size = 14),
    plot.margin = grid::unit(c(0, 0, 0, 0), "pt")
  )
legend <- cowplot::get_legend(legend_plot)

# 10) Arrange maps and legend into a final plot
maps_2x2 <- plot_grid(mg_map, es_map, rj_map, sp_map, ncol = 2, align = "hv")
final_plot <- plot_grid(maps_2x2, legend, ncol = 2, rel_widths = c(1, 0.14))

# 11) Save the final plot as a PDF
ggsave(
  "C:/Users/PC GAMER/Downloads/dataset/results/southeast_risk_maps.pdf",
  final_plot,
  width = 16, height = 10, device = cairo_pdf
)

# 12) Create a summary CSV
vars_numeric <- clusters %>%
  select(where(is.numeric), -any_of(c("prob", "cluster"))) %>%
  names()

topN_if_none <- 15

top20_all_states <- map_dfr(
  .x = 1:nrow(state_info),
  .f = function(i) {
    code <- state_info$code_state[i]
    uf <- state_info$state[i]
    df_st <- map_sudeste %>%
      filter(code_state == code, !is.na(prob)) %>%
      st_drop_geometry()
    
    if (nrow(df_st) == 0) return(NULL)
    
    vars_state <- intersect(vars_numeric, names(df_st))
    
    vars_with_variance <- vars_state[sapply(df_st[vars_state], function(col) {
      if (all(is.na(col))) return(FALSE)
      sd(col, na.rm = TRUE) > 0
    })]
    
    if (length(vars_with_variance) == 0) return(NULL)
    
    corrs <- sapply(vars_with_variance, function(v) cor(df_st[[v]], df_st$prob, use = "complete.obs"))
    corrs_df <- tibble(variable = names(corrs), correlation = corrs) %>%
      arrange(desc(abs(correlation))) %>%
      slice_head(n = 20)
    
    muni_risk <- df_st %>% filter(risk_cat == "Very High")
    if (nrow(muni_risk) == 0) {
      muni_risk <- df_st %>%
        arrange(desc(prob)) %>%
        slice_head(n = topN_if_none)
    }
    
    muni_risk <- muni_risk %>%
      select(cod6, cluster, prob, risk_cat) %>%
      mutate(state = uf) %>%
      arrange(desc(prob)) %>%
      mutate(rank = row_number())
    
    muni_cross <- merge(muni_risk, corrs_df, all = TRUE) %>% na.omit()
    muni_cross %>% select(state, cod6, cluster, prob, rank, risk_cat, variable, correlation)
  }
)

# Write the final data to CSV
write_csv(
  top20_all_states,
  "C:/Users/PC GAMER/Downloads/dataset/results/southeast.csv"
)

