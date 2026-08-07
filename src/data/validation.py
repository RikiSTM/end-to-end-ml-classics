import great_expectations as gx


def validate_raw(df):
    
    # 1. Initialize Context
    context = gx.get_context()
    
    # 2. Setup Data Source & Asset
    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="pandas_asset")
    
    # 3. Build BATCH DEFINITION (Ini yang diminta oleh error Pydantic tadi)
    batch_definition = data_asset.add_batch_definition_whole_dataframe("pandas_batch_def")
    
    # 4. Create Expectation Suite
    suite = gx.ExpectationSuite(name="pandas_suite")
    
    # 5. Define Rules
    required_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", 
        "Contract", "PaymentMethod", "Churn"
    ]
    for col in required_cols:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="TotalCharges"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Contract", 
        value_set=["Month-to-month", "One year", "Two year"]
    ))
    
    suite = context.suites.add(suite)
    
    # 6. Validation Definition (Masukkan batch_definition ke sini)
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="pandas_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    
    # 7. RUN! (DataFrame disuntikkan di saat eksekusi)
    results = validation_definition.run(batch_parameters={"dataframe": df})
    
    # 8. Evaluate
    if not results.success:
        failed_expectations = [
            res.expectation_config.kwargs 
            for res in results.results if not res.success
        ]
        raise ValueError(f"Data validation failed! Details: {failed_expectations}")

    print("✅ Gate 1: GX Raw data validation passed successfully via ValidationDefinition")


def validate_sql_layer():
    """
    Showcase: Validating data directly inside SQLite using Great Expectations Query Generator
    """
    context = gx.get_context()
    
    connection_string = "sqlite:///data/database/churn.db"
    data_source = context.data_sources.add_sqlite("churn_db_source", connection_string=connection_string)
    data_asset = data_source.add_table_asset(name="customers_raw_asset", table_name="customers_raw")
    
    # Batch Definition khusus untuk SQL Table
    batch_definition = data_asset.add_batch_definition_whole_table("sql_batch_def")
    
    suite = gx.ExpectationSuite(name="sql_suite")
    
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="TotalCharges"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]))
    
    suite = context.suites.add(suite)
    
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="sql_validation",
            data=batch_definition,
            suite=suite,
        )
    )
    
    # RUN untuk SQL tidak perlu parameter dataframe
    results = validation_definition.run()
    
    if not results.success:
        failed_expectations = [
            res.expectation_config.kwargs 
            for res in results.results if not res.success
        ]
        raise ValueError(f"SQL Database Quality Gate Failed! Details: {failed_expectations}")
        
    print("✅ Gate 2: GX SQL-level validation passed successfully via Query Pushdown")