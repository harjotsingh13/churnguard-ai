def calculate_ltv_revenue_at_risk(
        probability,
        monthly_charge,
        tenure,
        churn_rate
):
    
    expected_lifetime = 1 / churn_rate
    
    remaining_lifetime = expected_lifetime - tenure
    
    remaining_lifetime = max(remaining_lifetime, 1)
    
    revenue_at_risk = (
        probability *
        remaining_lifetime *
        monthly_charge
    )
    
    return revenue_at_risk



def recommend_action(probability, contract, tenure, tech_support):

    # High Risk Customers
    if probability >= 0.7:
        if contract == "Month-to-month":
            return "Offer 10% retention discount and convert to yearly contract"

        if tech_support == "No":
            return "Provide free premium tech support trial"

        return "Assign retention specialist call"

    # Medium Risk Customers
    if 0.4 <= probability < 0.7:
        if tenure < 6:
            return "Send onboarding engagement email"

        return "Offer loyalty reward points"

    # Low Risk Customers
    return "No immediate action required"
