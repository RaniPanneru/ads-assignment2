# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 04:33:20 2023

@author: panne
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(filename):
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns
    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def calculate_summary_stats(df_years, countries, indicators):
    # create a dictionary to store the summary statistics
    summary_stats = {}

    # calculate summary statistics for each indicator and country
    for indicator in indicators:
        for country in countries:
            # summary statistics for individual countries
            stats = df_years.loc[(country, indicator)].describe()
            summary_stats[f'{country} - {indicator}'] = stats

        # summary statistics for the world
        stats = df_years.loc[('World', indicator)].describe()
        summary_stats[f'World - {indicator}'] = stats

    return summary_stats


def print_summary_stats(summary_stats):
    # print the summary statistics
    for key, value in summary_stats.items():
        print(key)
        print(value)
        print()


# create scatter plots

def create_scatter_plots(df_years, indicators, countries):
    
        
    for country in countries:
         for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                x = df_years.loc[(country, indicators[i])]
                y = df_years.loc[(country, indicators[j])]
                plt.scatter(x, y)
                plt.xlabel(indicators[i])
                plt.ylabel(indicators[j])
                plt.title(f'{country} ')
                plt.show()


def subset_data(df_years, countries, indicators):
    df = df_years.loc[(countries, indicators), :]
    df = df.transpose()
    return df


def calculate_correlations(df):
    corr = df.corr()
    return corr


def visualize_correlations(corr):
    sns.heatmap(corr, cmap='coolwarm', annot=True, square=True)
    plt.title('Correlation Matrix of Indicators')
    plt.show()
    
def create_pivot_chart(df_years, indicators, countries, selected_years):
    df_pivot = df_years.loc[(countries, indicators), selected_years]
    df_pivot = df_pivot.transpose()

    # Plotting the pivot chart
    df_pivot.plot(kind='bar', stacked=True)
    plt.xlabel('Year')
    plt.ylabel(', '.join(indicators))
    plt.title(f'Urban Population Indicators Over Time')
    plt.legend(title='Country', bbox_to_anchor=(1, 1))
    plt.show()

def plot_area_plot(df_years, indicator, countries, years):
    for country in countries:
        indicator_values = df_years.loc[(country, indicator), years].dropna().values
        plt.fill_between(years, indicator_values, label=country)

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'{indicator} Over Time')
    plt.legend()
    plt.show()

def plot_bar_graph(df_years, indicator, countries, years):
    for year in years:
        
        x = np.arange(len(countries))
        width = 0.35

    for year in years:
        fig, ax = plt.subplots()
        indicator_values = []
        for country in countries:
            indicator_values.append(df_years.loc[(country, indicator), year])

        rects = ax.bar(x, indicator_values, width, label=str(year))

        ax.set_xlabel('Country')
        ax.set_ylabel('Value')
        ax.set_title(f'{indicator} over the year in {year}')
        ax.set_xticks(x)
        ax.set_xticklabels(countries)
        ax.legend()

        fig.tight_layout()
        plt.show()



def plot_heatmap(df, title):
    df_last_5_years = df.iloc[:, -5:]  # Extract data for the last 5 years
    sns.heatmap(df_last_5_years.corr(), cmap='YlGnBu', center=0, annot=True, linewidths=0.05)
    plt.title(title + ' - Last 5 Years')
    plt.figure(dpi=300)
    plt.show()




def plot_line_plot(df_years, indicator, countries):
    for country in countries:
        indicator_values = df_years.loc[(country, indicator), :].dropna().values
        plt.plot(df_years.columns, indicator_values, label=country)

    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'{indicator} Over Time')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df_years, df_countries = read_data(
        r"C:\Users\panne\Downloads\dataname.csv")
    urban_population_indicators = [
        'Urban population (% of total population)',
        'Urban population growth (annual %)']
    # Selecting two bar graphs
    plot_bar_graph(df_years, 'Urban population (% of total population)', ['United States', 'China', 'India', 'Japan'], [ 2010, 2020])
   
    
    # Selecting three scatter plots
    create_scatter_plots(df_years, ['Urban population (% of total population)', 'Urban population growth (annual %)'], ['United States', 'China', 'India'])
    
    plot_area_plot(df_years, 'Urban population', [ 'China', 'India', 'Japan'], np.arange(2000, 2021))
    # Selecting two heatmaps
    df_heatmap_1 = subset_data(df_years, ['United States', 'China', 'India', 'Japan'], ['Urban population (% of total population)', 'Urban population growth (annual %)'])
    plot_heatmap(df_heatmap_1, 'Urban Population Indicators')

    df_heatmap_2 = subset_data(df_years, ['United States', 'China', 'India', 'Japan'], ['Urban land area where elevation is below 5 meters (sq. km)', 'Population in urban agglomerations of more than 1 million (% of total population)'])
    plot_heatmap(df_heatmap_2, 'Urban Land Area and Population Distribution')

    # Selecting a line plot
    plot_line_plot(df_years, 'Urban population', ['United States', 'China', 'India'])
    
    
    selected_years = [2000, 2010, 2020]
    create_pivot_chart(df_years, urban_population_indicators, ['United States', 'China', 'India','japan'], selected_years)