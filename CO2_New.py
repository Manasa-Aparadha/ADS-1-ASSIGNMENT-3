import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import curve_fit
from errors import err_ranges


# Load the dataset
df = pd.read_csv("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv", skiprows=4)

# Take a look at the data
df.head()

df.shape


# Load the dataset
df = pd.read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5447781.csv", skiprows=4)

# Select relevant columns and drop missing values
df = df[["Country Name", "2019"]].dropna()

# Rename columns
df.columns = ["Country", "GDP_per_capita"]

# Set index to country name
df.set_index("Country", inplace=True)

# Remove rows with invalid GDP values (negative or zero)
df = df[df["GDP_per_capita"] > 0]

# Log-transform the GDP values to reduce skewness
df["GDP_per_capita"] = np.log(df["GDP_per_capita"])

# Standardize the data using z-score normalization
df = (df - df.mean()) / df.std()

# Save the cleaned dataset to a new file
df.to_csv("cleaned_dataset.csv")


# load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# extract GDP per capita column and normalize
X = df['GDP_per_capita'].values.reshape(-1,1)
X_norm = (X - X.mean()) / X.std()

# range of number of clusters to try
n_clusters_range = range(2, 11)

# iterate over number of clusters and compute silhouette score
silhouette_scores = []
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_norm)
    silhouette_scores.append(silhouette_score(X_norm, labels))

# plot silhouette scores
plt.plot(n_clusters_range, silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal Number of Clusters")
plt.show()


# load cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# perform k-means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df[['GDP_per_capita']])
df['Cluster'] = kmeans.labels_

# plot results
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(df.index, df['GDP_per_capita'], c=df['Cluster'], cmap='Set1')
plt.xticks(np.arange(0, df.shape[0], 50), np.arange(0, df.shape[0], 50), fontsize=12)
plt.xlabel('Country Index', fontsize=14)
plt.ylabel('GDP per capita', fontsize=14)
plt.title('K-Means Clustering Results', fontsize=16)

# add legend for the cluster
legend = ax.legend(*scatter.legend_elements(), title='Cluster', fontsize=12)
ax.add_artist(legend)

# add annotation for the cluster centers
centers = kmeans.cluster_centers_
for i, center in enumerate(centers):
    ax.annotate(f'Cluster {i+1} center: {center[0]:,.2f}', xy=(1, center[0]), xytext=(6, 0), 
                textcoords="offset points", ha='left', va='center', fontsize=12, color='black')

plt.show()


# print countries in each cluster
for i in range(3):
    print(f'Cluster {i+1}:')
    print(df[df['Cluster']==i]['Country'].values)
    print('\n')

# evaluate silhouette score
silhouette_avg = silhouette_score(df[['GDP_per_capita']], kmeans.labels_)
print(f'Silhouette score: {silhouette_avg:.2f}')

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv", skiprows=4)

# Remove unnecessary columns
df = df.drop(columns=["Indicator Name", "Indicator Code", "Unnamed: 66"])

# Rename the remaining columns
df = df.rename(columns={"Country Name": "country", "Country Code": "code", "2015": "co2_per_capita"})

# Define the list of G7 country codes
g7_codes = ["CAN", "FRA", "DEU", "ITA", "JPN", "GBR", "USA"]

# Filter the dataframe to include only G7 countries
df = df[df["code"].isin(g7_codes)]

# Plot the carbon dioxide emissions per capita for each G7 country over time
for i, row in df.iterrows():
    code = row["code"]
    country = row["country"]
    data = row[-10:-1].astype(float)
    plt.plot(data, label=country)

plt.xlabel("Year")
plt.ylabel("CO2 emissions per capita (metric tons)")
plt.legend()
plt.show()


# Load the dataset into a pandas DataFrame
df = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv', skiprows=4)

# Select only the necessary data for fitting analysis
df = df[['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', *df.columns[-30:-1]]]

# Rename columns to simpler names
df.columns = ['Country', 'Code', 'Indicator', 'IndicatorCode', *range(1990, 2019)]

# Melt the DataFrame to transform the columns into rows
df_melted = pd.melt(df, id_vars=['Country', 'Code', 'Indicator', 'IndicatorCode'], var_name='Year', value_name='Value')

# Drop rows with missing values
df_cleaned = df_melted.dropna()

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('newfit_data.csv', index=False)


# Load the data
df = pd.read_csv("newfit_data.csv")

# Select data for the United Kingdom from 1990 to 2019
uk_data = df[(df['Country'] == 'United Kingdom') & (df['Year'] >= 1990) & (df['Year'] <= 2019)]

# Plot a histogram of the data
plt.hist(uk_data['Value'], bins=20)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of United Kingdom Data from 1990 to 2019")
plt.show()


# Load the data
data = pd.read_csv("newfit_data.csv")

# Select data only for United Kingdom
data_uk = data[data['Country'] == 'Kenya']

# Define a logistic function
def logistic(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))

# Fit the model to the data
x = data_uk["Year"]
y = data_uk["Value"]

popt, pcov = curve_fit(logistic, x, y)

# Define the range of x values for predictions
x_pred = np.arange(x.min(), x.max() + 20)

# Calculate the predicted y values and error ranges
y_pred = logistic(x_pred, *popt)
lower, upper = err_ranges(x_pred, logistic, popt, [1, 2, 3])

# Plot the model and the data
plt.figure(figsize=(12,6))
plt.plot(x, y, ".", label="CO2 emissions per capita (metric tons)", color='blue')
plt.plot(x_pred, y_pred, label="Line of Best Fit", color='green')
plt.fill_between(x_pred, lower, upper, alpha=0.3, color='gray', label="95% Confidence Range")
plt.xlabel("Year", fontsize=12)
plt.ylabel("CO2 emissions per capita (metric tons)", fontsize=12)
plt.title("CO2 emissions per capita in relation to time for sample country")
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
plt.show()



def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of 
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
        
    The function does not have a plt.show() at the end so that the user 
    can savethe figure.
    """


    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm', location="bottom")
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.colorbar()
    # no plt.show() at the end
    
    
def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


def get_diff_entries(df1, df2, column):
    """ Compares the values of column in df1 and the column with the same 
    name in df2. A list of mismatching entries is returned. The list will be
    empty if all entries match. """

    
    # merge dataframes keeping all rows
    df_out = pd.merge(df1, df2, on=column, how="outer")
    print("total entries", len(df_out))
    # merge keeping only rows in common
    df_in = pd.merge(df1, df2, on=column, how="inner")
    print("entries in common", len(df_in))
    df_in["exists"] = "Y"

    # merge again
    df_merge = pd.merge(df_out, df_in, on=column, how="outer")

    # extract columns without "Y" in exists
    df_diff = df_merge[(df_merge["exists"] != "Y")]
    diff_list = df_diff[column].to_list()

    return diff_list
