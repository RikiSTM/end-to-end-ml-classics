def validate_raw(df):

    # 1. Schema Check
    required_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "Churn"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # 2. null check (critical)
    if df["TotalCharges"].isna().any():
        raise ValueError("TotalCharges contains NaN")

    # 3. range check
    if (df["tenure"] < 0).any():
        raise ValueError("Invalid tenure < 0")

    if (df["MonthlyCharges"] <= 0).any():
        raise ValueError("Invalid MonthlyCharges <= 0")

    # 4. categorical constraint
    allowed_churn = {"Yes", "No"}
    if not set(df["Churn"].unique()).issubset(allowed_churn):
        raise ValueError("Invalid Churn values")

    print("Raw data validation passed")