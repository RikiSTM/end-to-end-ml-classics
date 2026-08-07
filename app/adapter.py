def build_full_input(data: dict) -> dict:
    full = {
        "customerID": "dummy",
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PaperlessBilling": "Yes",
        **data
    }
# FIX TYPE
    full["TotalCharges"] = float(full["TotalCharges"])
    full["MonthlyCharges"] = float(full["MonthlyCharges"])
    full["tenure"] = int(full["tenure"])
    full["SeniorCitizen"] = int(full["SeniorCitizen"])
    return full
