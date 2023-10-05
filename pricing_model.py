from datetime import datetime


def price_contract(
    injection_dates, withdrawal_dates, date_price_series,
    injection_rate, withdrawal_rate, max_volume, storage_cost_per_month,
    injection_withdrawal_cost_per_mmbtu, transportation_cost_per_transaction
):
    total_cost = 0.0
    total_revenue = 0.0

    # Calculate injection costs
    for date in injection_dates:
        gas_price = date_price_series[date] * injection_rate
        inj_cost = injection_rate * injection_withdrawal_cost_per_mmbtu
        trans_cost = transportation_cost_per_transaction
        total_cost += gas_price + inj_cost + trans_cost

        # Debugging statements
        print(
            f"[Injection on {date}] Gas Price: ${gas_price:,.2f}, Injection Cost: ${inj_cost:,.2f}, Transport Cost: ${trans_cost:,.2f}")

    # Calculate withdrawal revenues
    for date in withdrawal_dates:
        revenue = date_price_series[date] * withdrawal_rate
        withdrawal_cost = withdrawal_rate * injection_withdrawal_cost_per_mmbtu
        trans_cost = transportation_cost_per_transaction
        total_revenue += revenue
        total_cost += withdrawal_cost + trans_cost

        # Debugging statements
        print(
            f"[Withdrawal on {date}] Revenue: ${revenue:,.2f}, Withdrawal Cost: ${withdrawal_cost:,.2f}, Transport Cost: ${trans_cost:,.2f}")

    # Calculate storage costs
    start_date = datetime.strptime(injection_dates[0], "%m/%d/%y")
    end_date = datetime.strptime(withdrawal_dates[-1], "%m/%d/%y")
    storage_duration = (end_date - start_date).days // 30
    storage_cost = storage_duration * storage_cost_per_month
    total_cost += storage_cost

    # Debugging statement for total costs and revenue
    print(
        f"Total Revenue: ${total_revenue:,.2f} | Total Costs: ${total_cost:,.2f}")

    return total_revenue - total_cost


def main():
    date_price_data = {
        "10/31/20": 10.1,
        "11/30/20": 10.3,
        "12/31/20": 11.0,
        "1/31/21": 10.9,
        "2/28/21": 10.9,
        "3/31/21": 10.9,
        "4/30/21": 10.4,
        "5/31/21": 9.84,
        "6/30/21": 10.0,
        "7/31/21": 10.1,
        "8/31/21": 10.3,
        "9/30/21": 10.2,
        "10/31/21": 10.1,
        "11/30/21": 11.2,
        "12/31/21": 11.4,
        "1/31/22": 11.5,
        "2/28/22": 11.8,
        "3/31/22": 11.5,
        "4/30/22": 10.7,
        "5/31/22": 10.7,
        "6/30/22": 10.4,
        "7/31/22": 10.5,
        "8/31/22": 10.4,
        "9/30/22": 10.8,
        "10/31/22": 11.0,
        "11/30/22": 11.6,
        "12/31/22": 11.6,
        "1/31/23": 12.1,
        "2/28/23": 11.7,
        "3/31/23": 12.0,
        "4/30/23": 11.5,
        "5/31/23": 11.2,
        "6/30/23": 10.9,
        "7/31/23": 11.4,
        "8/31/23": 11.1,
        "9/30/23": 11.5,
        "10/31/23": 11.8,
        "11/30/23": 12.2,
        "12/31/23": 12.8,
        "1/31/24": 12.6,
        "2/29/24": 12.4,
        "3/31/24": 12.7,
        "4/30/24": 12.1,
        "5/31/24": 11.4,
        "6/30/24": 11.5,
        "7/31/24": 11.6,
        "8/31/24": 11.5,
        "9/30/24": 11.8
    }

    sample_injection_dates = ["6/30/23", "7/31/23"]
    sample_withdrawal_dates = ["11/30/23", "12/31/23"]

    INJECTION_RATE = 1e6
    WITHDRAWAL_RATE = 1e6
    MAX_VOLUME = 2e6
    STORAGE_COST = 100e3
    INJECTION_WITHDRAWAL_COST = 0.01 # Corrected the cost value
    TRANSPORT_COST = 50e3

    value = price_contract(
        sample_injection_dates, sample_withdrawal_dates, date_price_data,
        INJECTION_RATE, WITHDRAWAL_RATE, MAX_VOLUME,
        STORAGE_COST, INJECTION_WITHDRAWAL_COST,
        TRANSPORT_COST
    )

    print(f"Value of the contract: ${value:,.2f}")


if __name__ == "__main__":
    main()