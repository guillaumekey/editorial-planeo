import streamlit as st
import pandas as pd
import numpy as np
import re

# Page Configuration
st.set_page_config(layout="wide", page_title="SEO Prioritization Tool")

# Custom CSS
st.markdown("""
<style>
    /* General styles */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e6e6e6;
    }

    /* Configuration table styles */
    .config-table {
        width: 100%;
        margin-bottom: 1rem;
    }
    .config-table th {
        background-color: #f0f2f6;
        padding: 8px;
        text-align: left;
        font-weight: 500;
    }
    .config-table td {
        padding: 8px;
        border-bottom: 1px solid #e6e6e6;
    }
    .config-table input {
        width: 100px !important;
    }

    /* Parameter container styles */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 1rem !important;
    }

    /* Label styles */
    .parameter-label {
        font-weight: 500;
        color: #333;
    }

    /* Statistics table styles */
    .metric-stats {
        margin: 1em 0;
    }
    .metric-stats th {
        background-color: #f0f2f6;
        font-weight: 600;
    }
    .metric-stats td, .metric-stats th {
        text-align: right !important;
        padding: 8px !important;
    }
    .metric-stats td:first-child, .metric-stats th:first-child {
        text-align: left !important;
    }
</style>
""", unsafe_allow_html=True)

# Column Mapping for CSV
COLUMN_MAPPING = {
    'Impressions': 'Impressions',
    'Clicks': 'Clicks',
    'Position': 'Position',
    'Inbound Links': 'Unique Inlinks'
}

# Default Metric Configurations
DEFAULT_METRICS = {
    'Impressions': {
        'min_0': 0, 'max_0': 250,
        'min_2_5': 250, 'max_2_5': 25000,
        'min_5': 25000, 'max_5': 1e8,
        'reverse': False
    },
    'Clicks': {
        'min_0': 0, 'max_0': 100,
        'min_2_5': 100, 'max_2_5': 500,
        'min_5': 500, 'max_5': 1e8,
        'reverse': False
    },
    'Position': {
        'min_0': 19, 'max_0': 100,
        'min_2_5': 2, 'max_2_5': 19,
        'min_5': 0, 'max_5': 2,
        'reverse': True
    },
    'Inbound Links': {
        'min_0': 0, 'max_0': 5,
        'min_2_5': 5, 'max_2_5': 15,
        'min_5': 15, 'max_5': 1e8,
        'reverse': False
    }
}


def create_score_distribution_data(df):
    """Create score distribution data with specific score ranges in exact order"""
    # Define fixed score ranges in desired order
    ranges = [
        {"min": 0, "max": 0, "label": "= 0", "order": 0},
        {"min": 0, "max": 2.5, "label": "> 0 and <= 2.5", "order": 1},
        {"min": 2.5, "max": 5, "label": "> 2.5 and <= 5", "order": 2},
        {"min": 5, "max": 10, "label": "> 5 and <= 10", "order": 3},
        {"min": 10, "max": 15, "label": "> 10 and <= 15", "order": 4},
        {"min": 15, "max": 20, "label": "> 15 and <= 20", "order": 5}
    ]

    distribution_data = []
    for range_def in ranges:
        if range_def["min"] == range_def["max"]:
            # For the "= 0" case
            count = len(df[df['Total_Score'] == range_def["min"]])
        else:
            count = len(df[
                            (df['Total_Score'] > range_def["min"]) &
                            (df['Total_Score'] <= range_def["max"])
                            ])

        distribution_data.append({
            "range": range_def["label"],
            "count": count,
            "order": range_def["order"]
        })

    # Sort by the predefined order
    return sorted(distribution_data, key=lambda x: x["order"])


def display_score_visualizations(filtered_df, min_score):
    """Display score distribution visualizations"""
    st.subheader("Score Distribution Analysis")

    # Create columns for the visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.write("URLs by Score Range")
        distribution_data = create_score_distribution_data(filtered_df)

        # Calculate statistics
        total_urls = len(filtered_df)
        urls_above_min = len(filtered_df[filtered_df['Total_Score'] >= min_score])
        urls_below_min = total_urls - urls_above_min

        # Convert to DataFrame and keep the order
        chart_data = pd.DataFrame(distribution_data)

        # Create a categorical type with ordered categories
        chart_data['range'] = pd.Categorical(
            chart_data['range'],
            categories=chart_data['range'].tolist(),
            ordered=True
        )

        # Sort and set index
        chart_data = chart_data.sort_values('order').set_index('range')

        # Display bar chart with fixed height and width
        st.bar_chart(
            data=chart_data['count'],
            height=400,
            use_container_width=True
        )

    with col2:
        st.write("Score Statistics")

        # Display key metrics
        col_stats1, col_stats2 = st.columns(2)
        col_stats1.metric("Total URLs", total_urls)
        col_stats1.metric(
            "URLs Above Min Score",
            urls_above_min,
            f"{urls_above_min / total_urls * 100:.1f}%"
        )
        col_stats2.metric(
            "URLs Below Min Score",
            urls_below_min,
            f"{urls_below_min / total_urls * 100:.1f}%"
        )

        # Display detailed statistics
        stats = filtered_df['Total_Score'].describe()
        stats_data = pd.DataFrame({
            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
            "Value": [
                f"{stats['mean']:.1f}",
                f"{stats['50%']:.1f}",
                f"{stats['std']:.1f}",
                f"{stats['min']:.1f}",
                f"{stats['max']:.1f}"
            ]
        })
        st.table(stats_data)


def display_instructions():
    """Displays instructions for using the tool"""
    st.markdown("""
    # 📋 How to Use This SEO Prioritization Tool

    ### Step 1: Screaming Frog Setup
    * Open Screaming Frog SEO Spider
    * Configure Google Search Console API:
        * Go to Configuration > API Access > Google Search Console
        * Connect your Google account
        * Select your property

    ### Step 2: Website Crawl
    * Enter your website URL in Screaming Frog
    * Start the crawl
    * Wait for completion

    ### Step 3: Export Data
    * Go to "Internal" tab in Screaming Frog
    * Click "Export" button
    * Choose "CSV" format
    * Save the file

    ### Step 4: Use This Tool
    * Upload your CSV file below
    * Adjust scoring parameters in the sidebar if needed
    * Review and download your prioritized results
    """)


def create_metric_table(metric_name, default_values):
    """Creates a configuration table for a metric"""
    st.write(f"Configuration {metric_name}")

    # Headers
    cols = st.columns([2, 1, 1])
    cols[0].write("**Range**")
    cols[1].write("**Min Value (>=)**")
    cols[2].write("**Max Value (<)**")

    # Score 0 Range
    cols = st.columns([2, 1, 1])
    cols[0].write("Score 0")
    min_0 = cols[1].number_input(
        f"{metric_name} Min 0",
        value=default_values.get('min_0', 0),
        key=f"{metric_name}_min_0",
        label_visibility="collapsed"
    )
    max_0 = cols[2].number_input(
        f"{metric_name} Max 0",
        value=default_values['max_0'],
        key=f"{metric_name}_max_0",
        label_visibility="collapsed"
    )

    # Score 2.5 Range
    cols = st.columns([2, 1, 1])
    cols[0].write("Score 2.5")
    min_2_5 = cols[1].number_input(
        f"{metric_name} Min 2.5",
        value=default_values['min_2_5'],
        key=f"{metric_name}_min_2_5",
        label_visibility="collapsed"
    )
    max_2_5 = cols[2].number_input(
        f"{metric_name} Max 2.5",
        value=default_values['max_2_5'],
        key=f"{metric_name}_max_2_5",
        label_visibility="collapsed"
    )

    # Score 5 Range
    cols = st.columns([2, 1, 1])
    cols[0].write("Score 5")
    min_5 = cols[1].number_input(
        f"{metric_name} Min 5",
        value=default_values['min_5'],
        key=f"{metric_name}_min_5",
        label_visibility="collapsed"
    )
    max_5 = cols[2].number_input(
        f"{metric_name} Max 5",
        value=default_values.get('max_5', 1e8),
        key=f"{metric_name}_max_5",
        label_visibility="collapsed"
    )

    return {
        (min_0, max_0): 0,
        (min_2_5, max_2_5): 2.5,
        (min_5, max_5): 5
    }


def load_data(uploaded_file):
    """Loads and filters data from CSV file"""
    df = pd.read_csv(uploaded_file)

    # Filter for indexable pages
    df = df[
        (df['Status Code'] == 200) &
        (df['Indexability'] == 'Indexable')
        ].copy()

    # Convert numeric columns and handle NaN values
    numeric_columns = {
        'Clicks': 0.0,
        'Impressions': 0.0,
        'Position': 0.0,
        'Unique Inlinks': 0
    }

    for col, default_value in numeric_columns.items():
        if col in df.columns:
            # Convert to numeric, replacing NaN with default value
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_value)

    # Print the first few rows of each metric in a more compact format
    st.write("Sample of metrics after loading:")

    # Create 4 columns for metrics display
    col1, col2, col3, col4 = st.columns(4)

    # Display each metric in its own column
    with col1:
        if 'Clicks' in df.columns:
            st.write("Clicks:")
            st.dataframe(df['Clicks'].head().reset_index().rename(columns={'index': '#'}), height=200)

    with col2:
        if 'Impressions' in df.columns:
            st.write("Impressions:")
            st.dataframe(df['Impressions'].head().reset_index().rename(columns={'index': '#'}), height=200)

    with col3:
        if 'Position' in df.columns:
            st.write("Position:")
            st.dataframe(df['Position'].head().reset_index().rename(columns={'index': '#'}), height=200)

    with col4:
        if 'Unique Inlinks' in df.columns:
            st.write("Unique Inlinks:")
            st.dataframe(df['Unique Inlinks'].head().reset_index().rename(columns={'index': '#'}), height=200)

    return df


def calculate_score(value, ranges, reverse=False):
    """Calculates score based on value ranges"""
    if pd.isna(value):
        return 0

    if not reverse:
        for (min_val, max_val), score in ranges.items():
            if min_val <= value < max_val:
                return score
    else:
        sorted_ranges = sorted(ranges.items(), key=lambda x: x[1])
        for (min_val, max_val), score in sorted_ranges:
            if min_val <= value < max_val:
                return score
    return 0


def calculate_distribution_stats(df, column):
    """Calculates distribution statistics for a column"""
    percentiles = [0, 25, 50, 75, 100]
    stats = df[column].describe(percentiles=[p / 100 for p in percentiles])

    return {
        'min': stats['min'],
        'max': stats['max'],
        'mean': stats['mean'],
        'median': stats['50%'],
        'percentiles': {p: stats[f'{p}%'] for p in percentiles}
    }


def display_metric_distribution(df, metrics_mapping):
    """Displays metrics distribution"""
    st.subheader("Metrics Distribution")

    stats_data = []
    for metric_name, metric_col in metrics_mapping.items():
        if metric_col in df.columns:
            stats = calculate_distribution_stats(df, metric_col)
            format_func = lambda x: f"{x:.1f}" if isinstance(x, float) else str(x)

            stats_data.append({
                'Metric': metric_name,
                'Minimum': format_func(stats['min']),
                'P25': format_func(stats['percentiles'][25]),
                'Median': format_func(stats['percentiles'][50]),
                'Mean': format_func(stats['mean']),
                'P75': format_func(stats['percentiles'][75]),
                'Maximum': format_func(stats['max'])
            })

    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df.style.set_properties(**{'text-align': 'right'}))

    st.info("""
        💡 **Threshold Setting Guide:**
        - Score 0: Consider values below P25 (above for Position)
        - Score 2.5: Consider values between P25 and P75
        - Score 5: Consider values above P75 (below for Position)
    """)


def apply_url_filters(df, include_patterns=None, exclude_patterns=None):
    """Applies regex filters to URLs"""
    filtered_df = df.copy()

    try:
        if include_patterns:
            pattern = '|'.join(f'(?:{p})' for p in include_patterns if p.strip())
            if pattern:
                filtered_df = filtered_df[
                    filtered_df['Address'].str.contains(pattern, case=False, regex=True, na=False)
                ]

        if exclude_patterns:
            pattern = '|'.join(f'(?:{p})' for p in exclude_patterns if p.strip())
            if pattern:
                filtered_df = filtered_df[
                    ~filtered_df['Address'].str.contains(pattern, case=False, regex=True, na=False)
                ]

    except re.error as e:
        st.error(f"Regex error: {str(e)}")
        return df

    return filtered_df


def main():
    """Main application function"""
    st.title("SEO Prioritization Tool")

    # Instructions in a collapsible section (collapsed by default)
    with st.expander("📋 How to Use This Tool", expanded=False):
        display_instructions()

    # Sidebar configuration
    with st.sidebar:
        st.header("Scoring Parameters")

        # URL filters section
        st.subheader("URL Filters")
        with st.expander("Filter Configuration", expanded=True):
            st.write("**Regex Filters**")

            # Include patterns
            st.write("Include URLs containing (one per line):")
            include_patterns = st.text_area(
                "Patterns to include",
                height=100,
                key="include_patterns",
                help="Enter regex patterns to include, one per line",
                label_visibility="collapsed"
            ).split('\n')

            # Exclude patterns
            st.write("Exclude URLs containing (one per line):")
            exclude_patterns = st.text_area(
                "Patterns to exclude",
                height=100,
                key="exclude_patterns",
                help="Enter regex patterns to exclude, one per line",
                label_visibility="collapsed"
            ).split('\n')

        # Metrics configuration
        metrics = {}
        for metric_name, default_values in DEFAULT_METRICS.items():
            with st.expander(f"{metric_name} Configuration", expanded=True):
                metrics[metric_name] = {
                    'ranges': create_metric_table(metric_name, default_values),
                    'reverse': default_values.get('reverse', False)
                }

    # Main section - Data Import
    st.header("Data Import")
    uploaded_file = st.file_uploader(
        "Upload Screaming Frog CSV file",
        type=['csv'],
        help="Limit 200MB per file • CSV"
    )

    if uploaded_file is not None:
        try:
            # Load and process data
            df = load_data(uploaded_file)

            # Display distribution statistics
            display_metric_distribution(df, COLUMN_MAPPING)

            # Calculate scores for each metric
            for metric, config in metrics.items():
                csv_column = COLUMN_MAPPING[metric]
                if csv_column not in df.columns:
                    st.error(f"Column {csv_column} not found in CSV file")
                    continue

                df[f'Score_{metric}'] = df[csv_column].apply(
                    lambda x: calculate_score(x, config['ranges'], config['reverse'])
                )

            # Calculate total score
            score_columns = [f'Score_{metric}' for metric in metrics.keys()]
            df['Total_Score'] = df[score_columns].sum(axis=1)

            # Apply URL filters first
            filtered_df = apply_url_filters(
                df,
                include_patterns=[p for p in include_patterns if p.strip()],
                exclude_patterns=[p for p in exclude_patterns if p.strip()]
            )

            # Minimum score slider
            max_possible_score = float(len(metrics.keys()) * 5)
            min_score = st.slider(
                "Minimum score",
                0.0, max_possible_score, 0.0,
                step=0.1
            )

            # Display score distribution visualizations with filtered data
            display_score_visualizations(filtered_df, min_score)

            # Results display
            st.header("Prioritization Results")

            # Define number formatting for different columns
            format_dict = {
                'Clicks': '{:.0f}',  # No decimals for Clicks
                'Impressions': '{:.0f}',  # No decimals for Impressions
                'Position': '{:.1f}',  # One decimal for Position
                'Total_Score': '{:.1f}',  # One decimal for scores
                'Score_Impressions': '{:.1f}',
                'Score_Clicks': '{:.1f}',
                'Score_Position': '{:.1f}',
                'Score_Inbound Links': '{:.1f}'
            }

            # Filter by score and sort
            results = filtered_df[filtered_df['Total_Score'] >= min_score].sort_values(
                'Total_Score',
                ascending=False
            )

            # Define display columns
            display_columns = [
                                  'Address', 'Title 1', 'Meta Description 1', 'H1-1', 'Word Count',
                                  'Crawl Depth', 'Clicks', 'Impressions', 'Position', 'Unique Inlinks',
                                  'Total_Score'
                              ] + score_columns

            # Function to apply color gradient to Total_Score
            def color_score(val):
                normalized_val = val / (len(metrics.keys()) * 5)
                return f'background-color: rgba(144, 238, 144, {normalized_val * 0.3})'

            # Style the dataframe
            styled_results = results[display_columns].style \
                .applymap(color_score, subset=['Total_Score']) \
                .format(format_dict)

            # Display number of results
            st.write(f"Number of URLs found: {len(results)}")

            # Display main results table with increased height
            st.dataframe(
                styled_results,
                use_container_width=True,
                height=600
            )

            # Display excluded URLs (URLs below minimum score)
            st.header("URLs to Clean")
            excluded_results = filtered_df[filtered_df['Total_Score'] < min_score].sort_values(
                'Total_Score',
                ascending=False
            )

            if not excluded_results.empty:
                styled_excluded = excluded_results[display_columns].style \
                    .applymap(color_score, subset=['Total_Score']) \
                    .format(format_dict)

                st.write(f"Number of URLs to clean: {len(excluded_results)}")
                st.dataframe(
                    styled_excluded,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No URLs below the minimum score threshold")

            # Add download buttons for both tables
            col1, col2 = st.columns(2)

            if not results.empty:
                with col1:
                    csv_results = results[display_columns].to_csv(index=False)
                    st.download_button(
                        "Download Prioritized URLs",
                        data=csv_results,
                        file_name="seo_prioritization_results.csv",
                        mime="text/csv"
                    )

            if not excluded_results.empty:
                with col2:
                    csv_excluded = excluded_results[display_columns].to_csv(index=False)
                    st.download_button(
                        "Download URLs to Clean",
                        data=csv_excluded,
                        file_name="seo_urls_to_clean.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Available columns in file:")
            if 'df' in locals():
                st.write(df.columns.tolist())


if __name__ == "__main__":
    main()