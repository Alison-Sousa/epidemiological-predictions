library(dplyr)
library(readr)

# Read the data
df <- read_csv("C:/Users/PC GAMER/Downloads/dataset/results/southeast.csv")

# Step 1: Get top 5 riskiest municipalities per state
top_munis_codes <- df %>%
  group_by(state) %>%
  # Sort by probability to find the highest risk
  arrange(desc(prob)) %>%
  # Get unique municipality codes
  distinct(cod6) %>%
  slice_head(n = 5) %>%
  ungroup()

# Step 2: For these municipalities, get top 10 correlated variables for each
panel_melhorado <- df %>%
  # Filter original data for the top municipalities from Step 1
  semi_join(top_munis_codes, by = c("state", "cod6")) %>%
  
  # Group by each individual municipality
  group_by(state, cod6, risk_cat, prob) %>%
  
  # Arrange variables by the strength of correlation (absolute value)
  arrange(desc(abs(correlation))) %>%
  
  # Get top 10 variables for EACH municipality
  slice_head(n = 10) %>%
  
  # Clean up and organize the final result
  ungroup() %>%
  mutate(correlation = round(correlation, 2)) %>%
  
  # 1. First, arrange the final table using 'prob' (which still exists here)
  arrange(state, desc(prob), desc(abs(correlation))) %>%
  
  # 2. Second, select the final columns (this removes 'prob' from the final view)
  select(state, cod6, risk_cat, variable, correlation)

# Print the result (showing more rows to see the structure)
print(panel_melhorado, n = 60)

# Save the new, more useful panel to a CSV file
write_csv(panel_melhorado, "C:/Users/PC GAMER/Downloads/dataset/results/panel_southeast.csv")

