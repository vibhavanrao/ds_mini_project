import pandas as pd
import numpy as np

# Load the updated dataset with churn
updated_dataset_path = "updated_dataset_with_churn.csv"
updated_dataset = pd.read_csv(updated_dataset_path)

# Add new features inspired by churn_input_data 2.csv

# 1. ARR Change Percentage
updated_dataset['arr_change_percentage'] = (
    (updated_dataset['future_arr'] - updated_dataset['current_arr']) / updated_dataset['current_arr']
) * 100

# 2. Log-transformed ARR
updated_dataset['log_current_arr'] = np.log1p(updated_dataset['current_arr'])
updated_dataset['log_future_arr'] = np.log1p(updated_dataset['future_arr'])

# 3. Customer Age in Years
updated_dataset['customer_age_years'] = updated_dataset['customer_age_quarters'] * 4

# 7. Severity Outages Ratio
updated_dataset['sev_outages_ratio'] = (
    updated_dataset['num_high_sev_outages'] / 
    (updated_dataset['num_low_sev_outages'] + 1)  # Adding 1 to avoid division by zero
)

# 8. Total Outages
updated_dataset['total_outages'] = (
    updated_dataset['num_low_sev_outages'] + updated_dataset['num_high_sev_outages']
)

# 9. Log-transformed Max Seats
updated_dataset['log_max_seats'] = np.log1p(updated_dataset['max_seats'])

# 10. CSAT Score Normalization
updated_dataset['csat_score_normalized'] = updated_dataset['csat_score'] / 100



# Save the updated dataset with new features
updated_dataset.to_csv("updated_dataset_with_new_features.csv", index=False)

print(f"Updated dataset with new features saved to updated_dataset_with_new_features.csv")