"""Shared input contract for training, API, form and CSV scoring."""

import math
import pandas as pd

# These fields are present in both source tables and the supplied unseen workbook.
FEATURES = {
    "NETMONTHLYINCOME": ("Net monthly income (source currency)", 0, 1000000000),
    "Time_With_Curr_Empr": ("Months with current employer", 0, 1200),
    "Tot_Missed_Pmnt": ("Missed payments", 0, 10000),
    "Age_Oldest_TL": ("Oldest account age (months)", 0, 1200),
    "Age_Newest_TL": ("Newest account age (months)", 0, 1200),
    "Secured_TL": ("Secured accounts", 0, 1000),
    "Unsecured_TL": ("Unsecured accounts", 0, 1000),
    "num_times_60p_dpd": ("Occurrences over 60 days past due", 0, 10000),
    "max_recent_level_of_deliq": ("Maximum recent delinquency (days)", 0, 10000),
    "num_std_12mts": ("Standard account observations in last 12 months", 0, 10000),
    "num_sub": ("Substandard account observations", 0, 10000),
    "num_dbt": ("Doubtful account observations", 0, 10000),
    "enq_L3m": ("Credit enquiries in last 3 months", 0, 1000),
    "time_since_recent_enq": ("Time since most recent enquiry (days)", 0, 100000),
}
LABELS = {
    "P1": "Best source category",
    "P2": "Second source category",
    "P3": "Third source category",
    "P4": "Last source category",
}


def validate_records(records):
    """Validate ordered numeric features; explicit nulls mean missing values."""
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("Provide a list of 1 to 500 records.")
    cleaned = []
    for row_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Row {row_number}: each record must be an object.")
        missing = set(FEATURES) - set(record)
        if missing:
            raise ValueError(
                f"Row {row_number}: missing columns: {', '.join(sorted(missing))}"
            )
        row = {}
        for field, (_, lower, upper) in FEATURES.items():
            value = record[field]
            if value is None or value == "" or value == -99999 or value == "-99999":
                row[field] = float("nan")
                continue
            try:
                if isinstance(value, bool):
                    raise ValueError
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {row_number}: {field} must be numeric.") from exc
            if not math.isfinite(number) or not lower <= number <= upper:
                raise ValueError(
                    f"Row {row_number}: {field} must be between {lower} and {upper}."
                )
            if field != "NETMONTHLYINCOME" and number != int(number):
                raise ValueError(f"Row {row_number}: {field} must be a whole number.")
            row[field] = number
        if all(math.isnan(value) for value in row.values()):
            raise ValueError(f"Row {row_number}: provide at least one observed value.")
        if (
            not math.isnan(row["Age_Oldest_TL"])
            and not math.isnan(row["Age_Newest_TL"])
            and row["Age_Newest_TL"] > row["Age_Oldest_TL"]
        ):
            raise ValueError(
                f"Row {row_number}: newest account cannot be older than oldest account."
            )
        cleaned.append(row)
    return pd.DataFrame(cleaned, columns=list(FEATURES))
