"""Generate realistic synthetic café sales data for testing and demonstration."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = datetime(2026, 4, 1)
NUM_DAYS = 92  # 3 months of data
PRODUCTS = {
    "Coffee": {"subproducts": ["Espresso", "Cappuccino", "Latte", "Americano"], "prices": [2.50, 3.50, 3.75, 2.75]},
    "Food": {"subproducts": ["Sandwich", "Salad", "Pastry", "Cake"], "prices": [6.50, 7.00, 3.50, 4.00]},
    "Drinks": {"subproducts": ["Smoothie", "Juice", "Water", "Soda"], "prices": [5.00, 4.50, 2.00, 3.00]},
}

def generate_hourly_base_sales(hour: int, day_of_week: int) -> float:
    """
    Base sales multiplier by hour of day and day of week.
    Afternoon dip (~2-3pm) and weekend patterns are built-in.
    """
    # Hourly pattern: low before 7am, peak at lunch (11-13), afternoon dip (14-15),
    # second peak at evening (17-19), low after 20pm
    hourly_pattern = {
        7: 0.3, 8: 0.6, 9: 0.9, 10: 1.2, 11: 1.5, 12: 1.4,  # morning ramp + lunch start
        13: 1.3, 14: 0.7, 15: 0.8,  # AFTERNOON DIP
        16: 1.0, 17: 1.3, 18: 1.2, 19: 0.9,  # evening
        20: 0.4, 21: 0.2,
    }
    # Default to very low for other hours
    base = hourly_pattern.get(hour, 0.05)

    # Weekend multiplier (Saturday=5, Sunday=6)
    if day_of_week in [5, 6]:
        base *= 1.15  # slightly higher on weekends

    return base

def generate_sales_data() -> pd.DataFrame:
    """Generate 92 days of synthetic café sales data."""
    records = []

    for day_offset in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday

        # Add noise and trends
        day_noise = np.random.normal(1.0, 0.15)  # day-to-day variation
        trend = 1.0 + (day_offset / NUM_DAYS) * 0.1  # slight upward trend

        # Simulate a dip in week 4 (e.g., staffing issue or weather)
        anomaly_mult = 1.0
        if 25 <= day_offset <= 30:
            anomaly_mult = 0.7

        for hour in range(7, 22):  # Café open 7am-9pm
            base_hourly = generate_hourly_base_sales(hour, day_of_week)
            hourly_multiplier = base_hourly * day_noise * trend * anomaly_mult

            # Randomize exact number of transactions
            num_transactions = np.random.poisson(max(2, int(20 * hourly_multiplier)))

            for _ in range(num_transactions):
                # Pick a random product
                category = np.random.choice(list(PRODUCTS.keys()), p=[0.45, 0.35, 0.20])
                product_idx = np.random.randint(0, len(PRODUCTS[category]["subproducts"]))
                product = PRODUCTS[category]["subproducts"][product_idx]
                price = PRODUCTS[category]["prices"][product_idx]

                # Quantity usually 1, sometimes 2
                quantity = np.random.choice([1, 2], p=[0.85, 0.15])

                records.append({
                    "date": current_date,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "category": category,
                    "product": product,
                    "quantity": quantity,
                    "unit_price": price,
                    "revenue": quantity * price,
                })

    df = pd.DataFrame(records)
    return df

def main():
    """Generate and save sample data."""
    df = generate_sales_data()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    output_path = os.path.join(os.path.dirname(__file__), "sample_sales.csv")
    df.to_csv(output_path, index=False)

    print(f"✓ Generated {len(df)} transactions across {df['date'].nunique()} days")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total revenue: £{df['revenue'].sum():,.2f}")
    print(f"  Saved to: {output_path}")

    return df

if __name__ == "__main__":
    main()
