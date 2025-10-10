library(sf)
library(geobr)
library(dplyr)
library(ggplot2)
library(readr)
library(stringr)

# 1. Load risk data
probs <- read_csv("C:/Users/PC GAMER/Downloads/dataset/data/R/prb.csv",
                  col_types = cols(
                    cod6 = col_character(),
                    municipio = col_character(),
                    year = col_integer(),
                    prob = col_double()
                  ))

# 2. Compute mean risk per municipality
probs_mean <- probs %>%
  group_by(cod6) %>%
  summarise(mean_risk = mean(prob, na.rm = TRUE))

# 3. Load municipality shapefile
map_muni <- read_municipality(code_muni = "all", year = 2022, simplified = TRUE) %>%
  mutate(cod6 = str_sub(as.character(code_muni), 1, 6)) %>%
  mutate(cod6 = as.character(cod6))

# 4. Simplify map geometry
map_muni <- st_simplify(map_muni, dTolerance = 1000, preserveTopology = TRUE)

# 5. Join risk data to map
map_data <- map_muni %>%
  left_join(probs_mean %>% mutate(cod6 = as.character(cod6)), by = "cod6") %>%
  filter(!is.na(mean_risk))

# 6. Create risk categories based on fixed probability thresholds (AJUSTE FEITO AQUI)
map_data <- map_data %>%
  mutate(
    risk_cat = cut(
      mean_risk,
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
      labels = c("Very Low", "Low", "Medium", "High", "Very High"),
      include.lowest = TRUE
    ),
    risk_cat = factor(
      risk_cat,
      levels = c("Very High", "High", "Medium", "Low", "Very Low")
    )
  )

# 7. Define color palette
colors <- c(
  "Very High" = "#d9534f",
  "High"      = "#f4b084",
  "Medium"    = "#ffe699",
  "Low"       = "#a9d18e",
  "Very Low"  = "#d4eac7"
)

# 8. Generate map
p <- ggplot(map_data) +
  geom_sf(aes(fill = risk_cat), color = NA) +
  scale_fill_manual(values = colors, name = "Risk") +
  theme_void() +
  theme(
    legend.position = "left",
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 12),
    plot.margin = grid::unit(c(0, 0, 0, 0), "pt")
  )

# 9. Save map to PDF
pdf("C:/Users/PC GAMER/Downloads/dataset/results/risk_map.pdf", width = 6, height = 5, useDingbats = FALSE)
print(p)
dev.off()

# 10. Count municipalities
total_municipalities <- nrow(map_data)
print(paste("Total de municípios:", total_municipalities))

# 11. Tabulate risk categories
print("Contagem de municípios por faixa de risco:")
print(table(map_data$risk_cat))

# 12. Define mapping for macro-regions (full names)
regions_full <- tibble(
  uf = c(
    "11","12","13","14","15","16","17",        # North
    "21","22","23","24","25","26","27","28","29", # Northeast
    "31","32","33","35",                        # Southeast
    "41","42","43",                            # South
    "50","51","52","53"                         # Central-West
  ),
  region = c(
    rep("North", 7),
    rep("Northeast", 9),
    rep("Southeast", 4),
    rep("South", 3),
    rep("Central-West", 4)
  )
)

# 13. Map municipalities to regions
map_data <- map_data %>%
  mutate(
    uf = str_sub(cod6, 1, 2)
  ) %>%
  left_join(regions_full, by = "uf")

# 14. Filter for high-risk municipalities
high_risk_full <- map_data %>%
  filter(risk_cat %in% c("High", "Very High"))

# 15. Count high-risk municipalities by region
region_counts_full <- high_risk_full %>%
  group_by(region) %>%
  summarise(qty = n()) %>%
  filter(!is.na(region)) %>%
  mutate(region = factor(region,
                         levels = c("North", "Northeast", "Central-West", "Southeast", "South")))

# 16. Create and save bar plot (full region names)
ggplot(region_counts_full, aes(x = region, y = qty, fill = region)) +
  geom_bar(stat = "identity", width = 0.7, show.legend = FALSE) +
  xlab("") + ylab("High-risk municipalities") +
  scale_fill_manual(values = c(
    "North" = "#8dd3c7",
    "Northeast" = "#ffffb3",
    "Central-West" = "#bebada",
    "Southeast" = "#fb8072",
    "South" = "#80b1d3"
  )) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(face = "bold", size = 13, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12),
        axis.title.y = element_text(face = "bold"),
        plot.title = element_blank())
ggsave("C:/Users/PC GAMER/Downloads/dataset/results/regions_full.pdf", width = 6, height = 5)


# 17. Define mapping for region abbreviations
regions_abbr <- tibble(
  uf = c(
    "11","12","13","14","15","16","17",        # N
    "21","22","23","24","25","26","27","28","29", # NE
    "31","32","33","35",                        # SE
    "41","42","43",                            # S
    "50","51","52","53"                         # CW
  ),
  region_abbr = c(
    rep("N", 7),
    rep("NE", 9),
    rep("SE", 4),
    rep("S", 3),
    rep("CW", 4)
  )
)

# 18. Map municipalities to region abbreviations
map_data <- map_data %>%
  left_join(regions_abbr, by = "uf")

# 19. Filter for high-risk municipalities
high_risk <- map_data %>%
  filter(risk_cat %in% c("High", "Very High"))

# 20. Count by region abbreviation
region_counts <- high_risk %>%
  group_by(region_abbr) %>%
  summarise(qty = n()) %>%
  filter(!is.na(region_abbr)) %>%
  mutate(region_abbr = factor(region_abbr,
                              levels = c("N", "NE", "CW", "SE", "S")))

# 21. Create and save bar plot (abbreviations)
ggplot(region_counts, aes(x = region_abbr, y = qty, fill = region_abbr)) +
  geom_bar(stat = "identity", width = 0.7, show.legend = FALSE) +
  xlab("") + ylab("High risk municipalities") +
  scale_fill_manual(values = c(
    "N" = "#8dd3c7",
    "NE" = "#ffffb3",
    "CW" = "#bebada",
    "SE" = "#fb8072",
    "S" = "#80b1d3"
  )) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(face = "bold", size = 16),
    axis.text.y = element_text(size = 12),
    axis.title.y = element_text(face = "bold"),
    plot.title = element_blank()
  )
ggsave("C:/Users/PC GAMER/Downloads/dataset/results/regions.pdf", width = 5, height = 4)

