import great_expectations as gx

def validate_raw(df):

    # Wrap the standard Pandas DataFrame into a Great Expectations Dataset object
    gdf = gx.from_pandas(df)
    
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
        gdf.expect_column_to_exist(col)

    # 2. Null Check: Ensure there are no missing values in critical numeric data
    gdf.expect_column_values_to_not_be_null("TotalCharges")

    # 3. Range Check: Ensure tenure values are non-negative
    gdf.expect_column_values_to_be_between("tenure", min_value=0)

    # 4. Value Constraint Check: Verify Churn contains only permitted categorical values
    gdf.expect_column_values_to_be_in_set("Churn", value_set=["Yes", "No"])
    
    # 5. Value Constraint Check: Verify Contract aligns with allowed business logic rules
    gdf.expect_column_values_to_be_in_set(
        "Contract", 
        value_set=["Month-to-month", "One year", "Two year"]
    )

    # Trigger and run all the declared expectations against the data
    results = gdf.validate()

    # Evaluate validation outcomes, acting as an assertion checkpoint
    if not results["success"]:
        # Parse the GX validation output JSON to isolate exactly what failed
        failed_expectations = [
            res["expectation_config"]["kwargs"] 
            for res in results["results"] if not res["success"]
        ]
        raise ValueError(f"Data validation failed! Details: {failed_expectations}")

    print("GX Raw data validation passed successfully")