import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def monte_carlo_simulation(n_runs, fund, n_investments,
                           vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                           growth_failure_rate, growth_min_return, growth_max_return,
                           growth_distribution_mean, growth_distribution_std):
    """
    Conduct Monte Carlo Simulation for Portfolio Returns
    
    Parameters:
    All simulation parameters
    
    Returns:
    tuple: Dataframe with raw data and summary statistics
    """
    data = []
    progress_bar = st.progress(0)

    # Simulation
    for n_growth in range(n_investments + 1):
        n_vc = n_investments - n_growth
        for _ in range(n_runs):
            # VC Investments
            vc_investments = np.maximum(np.random.power(vc_power_law_exponent, n_vc) - vc_failure_rate, 0)
            vc_investments *= np.random.uniform(vc_min_return, vc_max_return, n_vc)

            # Growth Investments
            a, b = (growth_min_return - growth_distribution_mean) / growth_distribution_std, (growth_max_return - growth_distribution_mean) / growth_distribution_std
            growth_investments = truncnorm.rvs(a, b, loc=growth_distribution_mean, scale=growth_distribution_std, size=n_growth)
            growth_investments *= (np.random.random(n_growth) > growth_failure_rate)

            # Total ROI
            total_roi = np.sum(vc_investments) + np.sum(growth_investments)
            data.append([n_growth, total_roi])
        
        progress_bar.progress(n_growth / n_investments)

    # Data Processing
    df = pd.DataFrame(data, columns=['growth_deals', 'roi'])
    df['roi'] = df['roi'] * fund / n_investments

    summary = df.groupby('growth_deals').roi.agg(['mean', 'std', 'count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
    summary.columns = ['growth_deals', 'mean_return', 'std_dev', 'count', 'median', 'percentile_25', 'percentile_75']

    return df, summary


def plot_data(summary, fund):
    """Plots the summary data."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(summary.growth_deals, summary.mean_return, label='Mean ROI', color='blue')
    ax.plot(summary.growth_deals, summary.percentile_25, label='25th Percentile', color='red')
    ax.plot(summary.growth_deals, summary.percentile_75, label='75th Percentile', color='green')
    ax.axhline(y=fund * 2, color='gray', linestyle='dashed')
    ax.axhline(y=fund * 3, color='gray', linestyle='dashed')
    ax.axhline(y=fund * 5, color='green', linestyle='dashed')
    ax.set_title('Monte Carlo Simulation of Portfolio Returns')
    ax.set_xlabel('Number of Growth Investments')
    ax.set_ylabel('Mean Return on Investment')
    ax.legend(['Mean ROI', '25th Percentile', '75th Percentile', '2x Fund', '3x Fund', '5x Fund'])
    st.pyplot(fig)


def main():
    st.title("Monte Carlo Simulation for Portfolio Returns")

    # Sidebar controls
    n_runs = st.sidebar.number_input("Number of simulations:", min_value=100, value=1000, step=100)
    fund = st.sidebar.number_input("Initial Fund:", min_value=100000, value=100000000, step=100000)
    n_investments = st.sidebar.number_input("Number of Investments:", min_value=1, value=20, step=1)

    st.sidebar.subheader("VC Investments")
    vc_failure_rate = st.sidebar.slider("VC Failure Rate:", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    vc_min_return = st.sidebar.number_input("VC Min Return Multiplier:", min_value=1.0, value=1.0, step=0.1)
    vc_max_return = st.sidebar.number_input("VC Max Return Multiplier:", min_value=1.0, value=25.0, step=0.1)
    vc_power_law_exponent = st.sidebar.slider("VC Power Law Exponent:", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

    st.sidebar.subheader("Growth Investments")
    growth_failure_rate = st.sidebar.slider("Growth Failure Rate:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    growth_min_return = st.sidebar.number_input("Growth Min Return Multiplier:", min_value=1.0, value=3.0, step=0.1)
    growth_max_return = st.sidebar.number_input("Growth Max Return Multiplier:", min_value=1.0, value=30.0, step=0.1)
    growth_distribution_mean = st.sidebar.slider("Growth Distribution Mean:", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
    growth_distribution_std = st.sidebar.slider("Growth Distribution Std Dev:", min_value=1.0, max_value=20.0, value=7.0, step=1.0)

    # Ensure max_return >= min_return
    vc_max_return = max(vc_max_return, vc_min_return)
    growth_max_return = max(growth_max_return, growth_min_return)

    # Run simulation
    st.subheader("Simulation Progress")
    data, summary = monte_carlo_simulation(n_runs, fund, n_investments,
                                           vc_failure_rate, vc_min_return, vc_max_return, vc_power_law_exponent,
                                           growth_failure_rate, growth_min_return, growth_max_return,
                                           growth_distribution_mean, growth_distribution_std)

    # Plotting
    st.subheader("Results")
    plot_data(summary, fund)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.table(summary)

    # Raw data
    st.subheader("Raw Data")
    st.write(data)


if __name__ == "__main__":
    main()
