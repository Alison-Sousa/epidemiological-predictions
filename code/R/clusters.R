# Load required libraries
library(sf)
library(geobr)
library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(patchwork)

# 1. Read clusters file
clusters <- read_csv("C:/Users/PC GAMER/Downloads/dataset/data/R/clu.csv",
                     col_types = cols(
                       cod6 = col_character(),
                       prob = col_double(),
                       clu = col_integer()
                     )) %>%
  rename(cluster_num = clu)

# MODIFICATION: Changed labels from "High", "Low", etc. to "Cluster 1", "Cluster 2", etc.
clusters <- clusters %>%
  mutate(cluster = factor(cluster_num, levels = c(0, 2, 3, 1),
                          labels = c("Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4")))


# 2. Load and simplify municipality shapes
muni_shapes <- read_municipality(year = 2022, simplified = TRUE) %>%
  mutate(cod6 = str_sub(as.character(code_muni), 1, 6),
         cod6 = as.character(cod6))
muni_shapes <- st_simplify(muni_shapes, dTolerance = 5000, preserveTopology = TRUE)

# 3. Join cluster data to map
clustered_map <- muni_shapes %>%
  left_join(clusters, by = "cod6") %>%
  filter(!is.na(cluster))

# 4. Color palette (used for the maps only now)
cluster_colors <- c(
  "Cluster 1" = "#d73027",
  "Cluster 2" = "#fc8d59",
  "Cluster 3" = "#91bfdb",
  "Cluster 4" = "#4575b4"
)

# --- 5, 6, 7. MAP PLOTTING ---
# 5. Function to plot each cluster map
plot_cluster <- function(cluster_name) {
  ggplot() +
    geom_sf(data = clustered_map, fill = "#eeeeee", color = NA) +
    geom_sf(data = filter(clustered_map, cluster == cluster_name),
            aes(fill = cluster), color = NA) +
    scale_fill_manual(values = cluster_colors[cluster_name], drop = FALSE, name = "Cluster") +
    labs(title = NULL) + # MODIFICATION: Removed plot titles
    theme_void() +
    theme(
      legend.position = "none",
      plot.margin = grid::unit(c(0, 0, 0, 0), "pt")
    )
}
# 6. Create individual maps
p_c1 <- plot_cluster("Cluster 1")
p_c2 <- plot_cluster("Cluster 2")
p_c3 <- plot_cluster("Cluster 3")
p_c4 <- plot_cluster("Cluster 4")
# 7. Combine the 4 maps into a 2x2 grid
p_all <- (p_c1 | p_c2) / (p_c3 | p_c4)
# Save maps to PDF
pdf("C:/Users/PC GAMER/Downloads/dataset/results/clusters_4maps.pdf", width = 10, height = 8, useDingbats = FALSE)
print(p_all)
dev.off()


# --- 8. VERTICAL BAR: NUMBER OF MUNICIPALITIES PER CLUSTER ---
# MODIFICATION: No title, labels in English, x-axis uses "Cluster 1", etc.
p_bar <- ggplot(clusters, aes(x = cluster)) +
  geom_bar(fill = "#d73027") +
  labs(
    title = NULL,
    x = "Cluster",
    y = "Number of Municipalities"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.title = element_text(face="bold"),
    axis.text.x = element_text(size=12)
  )

pdf("C:/Users/PC GAMER/Downloads/dataset/results/instances.pdf", width=8, height=6, useDingbats=FALSE)
print(p_bar)
dev.off()


# --- 9. HORIZONTAL BAR: PERCENTAGE OF MUNICIPALITIES PER CLUSTER ---
cluster_pct <- clusters %>%
  group_by(cluster) %>%
  summarise(n = n()) %>%
  mutate(pct = n / sum(n) * 100)

# MODIFICATION: No title, labels in English, and all bars are the same red color.
p_pct <- ggplot(cluster_pct, aes(x = pct, y = reorder(cluster, pct))) +
  geom_col(fill = "#d73027") + # All bars set to the same color
  geom_text(aes(label = sprintf("%.1f%%", pct)), hjust = -0.2, size = 4.5) +
  labs(
    title = NULL,
    x = "Percentage (%)",
    y = "Cluster"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.title = element_text(face="bold"),
    axis.text.y = element_text(size=12)
  ) +
  xlim(0, max(cluster_pct$pct) * 1.15) # Add extra space for labels

pdf("C:/Users/PC GAMER/Downloads/dataset/results/percentual_instances.pdf", width=9, height=6, useDingbats=FALSE)
print(p_pct)
dev.off()

print(table(clusters$cluster))
