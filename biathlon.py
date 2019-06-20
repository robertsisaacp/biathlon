#%% [markdown]
# # Biathlon Data Science Project
# 
# * Data source: Real Biatholon Statistics [link](http://www.realbiathlon.com/p/statistics-women.html)
# * 2015-2016 Season Biathlon World Cup Womens Shooting Times: [link](http://realbiathlon.sportsontheweb.net/data2/WomenShootingTimes_files/sheet016.htm)
# * 2015-2016 Season Biathlon World Cup Womens Results: [link](http://realbiathlon.sportsontheweb.net/data2/WomenResults_files/sheet016.htm)
# 
# ### Definition of Features
# 
# From the metadata on the data source we have the following definition of the features.
# 
# | Feature  | Description |
# |--------------|-------------|
# |  Family Name  | |
# |  Given Name  | |
# |  Nation  | |
# |  Races  |  Finished individual, non-relay races|
# |  Sh_Time_P |  Average Shooting Time in individual, non-relay races - Prone|
# |  Sh_Time_S |  Average Shooting Time in individual, non-relay races - Standing|
# |  Sh_Time_T |  Average Shooting Time in individual, non-relay races - Total|
# |  Sh_Time_IN |  Average Shooting Time in individual, non-relay races - Individual|
# |  Sh_Time_SP |  Average Shooting Time in individual, non-relay races - Sprint|
# |  Sh_Time_PU |  Average Shooting Time in individual, non-relay races - Pursuit|
# |  Sh_Time_MS |  Average Shooting Time in individual, non-relay races - Mass Start|
# |  Sh_Time_P |  Average Shooting Time in individual, non-relay races - Prone|
# |  Rg_Time_S |  Average Range Time in individual, non-relay races - Standing|
# |  Rg_Time_T |  Average Range Time in individual, non-relay races - Total|
# |  Rg_Time_IN |  Average Range Time in individual, non-relay races - Individual|
# |  Rg_Time_SP |  Average Range Time in individual, non-relay races - Sprint|
# |  Rg_Time_PU |  Average Range Time in individual, non-relay races - Pursuit|
# |  Rg_Time_MS |  Average Range Time in individual, non-relay races - Mass Start|
# |  Penalty Loop  |  Season average penalty loop (approximation)|
# 
# ### What we can do next
# * Merge shooting time dataframe with results dataframe to create a target for training our model
#     * We can do this by creating a unique key using first and last name and reindex by this key
# * Statistical analysis -- what is the spread of each category? Vizualize this
# * Feature selection -- What categories might contribute more to the final ranking? ("Skiing is for show, shooting is for dough")
# * Train a model to predict placement. How accurate is it? Would combining data from different years yield better results?
# * How have the categories changed throughout the years? Is the sport getting more competitive?
#%%
import pandas as pd
import matplotlib.pyplot as plt      # matplotlib.pyplot plots data
import seaborn as sns
import datetime

#%% [markdown]
# The dataset already contains an index column. We can either set this column as the index in our data frame, our we can choose to drop the first column in our dataset. I am doing the former below. Further, we see a large number of NaN's. Let's go ahead and convert these to 0:00.0 to be consistent. While we're at it, let's also rename the columns to something less confusing.

#%%
df_sh_times = pd.read_csv("shooting_time.csv", index_col=0)
df_sh_times = df_sh_times.fillna('0:00.0')
df_sh_times = df_sh_times.rename(columns={'Sh Time\nP': 'Sh_Time_P', 
                        'Sh Time\nS': 'Sh_Time_S',
                        'Sh Time\nT': 'Sh_Time_T',
                        'Rg Time\nP': 'Rg_Time_P',
                        'Rg Time\nS': 'Rg_Time_S',
                        'Rg Time\nT': 'Rg_Time_T',
                        'Sh Time\nIN': 'Sh_Time_IN',
                        'Sh Time\nSP': 'Sh_Time_SP',
                        'Sh Time\nPu': 'Sh_Time_PU',
                        'Sh Time\nMS': 'Sh_Time_MS',
                        'Rg Time\nIN': 'Rg_Time_IN',
                        'Rg Time\nSP': 'Rg_Time_SP',
                        'Rg Time\nPU': 'Rg_Time_PU',
                        'Rg Time\nMS': 'Rg_Time_MS',
                        'Penalty\nLoop': "Penalty Loop"})
df_sh_times.head()

#%% [markdown]
# We see that the times are stored as string. Let's write a function to convert each of the times to total seconds.

#%%
def str_to_sec(val):
    minute, seconds = val.split(":")
    sec, msec = seconds.split(".")
    total_sec = int(minute)*60 + int(sec) + int(msec)/10
    return total_sec


#%%
#region test
for c in df_sh_times.columns[4::]:
    df_sh_times[c] = df_sh_times[c].apply(lambda x: str_to_sec(x))
    
df_sh_times.head()

#endregion
#%% [markdown]
# Much better! Now that the data is in a more workable format, let's dive deeper into what the dataset looks like. Some things we might be interested in include the shape, a summary of each column (number of entries, data types etc), and a correlation matrix to pick out any redundancies.

#%%
# Look at the shape of the dataframe (number of rows and columns)
print("This data set has {rows} rows and {cols} columns".format(rows=df_sh_times.shape[0], cols=df_sh_times.shape[1]))


#%%
# look a a summary of each column (number of entries, data-type etc)
df_sh_times.info()


#%%
# Check for any null values
df_sh_times.isnull().values.any()


#%%
def plot_pretty_corr(df):
    plt.figure(figsize=(12, 12))
    corr = df.corr()
    ax = sns.heatmap(corr,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );


#%%
plot_pretty_corr(df_sh_times)


#%%
# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution
df_sh_times.describe()


#%%
# Vizualize using box plots
plt.figure(figsize=(10,10))
plt.boxplot([df_sh_times["Sh_Time_P"],
             df_sh_times["Sh_Time_S"], 
             df_sh_times["Sh_Time_T"],
             df_sh_times["Sh_Time_IN"],
             df_sh_times["Sh_Time_SP"],
             df_sh_times["Sh_Time_PU"],
             df_sh_times["Sh_Time_MS"],
            ],
            labels = ["Sh_Time_P", "Sh_Time_S", "Sh_Time_T",
                      "Sh_Time_IN", "Sh_Time_SP", "Sh_Time_PU", "Sh_Time_MS"])
plt.ylabel("Seconds")
plt.title("Shooting Time Distribution (seconds)")
plt.show()


#%%
# Vizualize using box plots
plt.figure(figsize=(10,10))
plt.boxplot([df_sh_times["Rg_Time_P"],
             df_sh_times["Rg_Time_S"], 
             df_sh_times["Rg_Time_T"],
             df_sh_times["Rg_Time_IN"],
             df_sh_times["Rg_Time_SP"],
             df_sh_times["Rg_Time_PU"],
             df_sh_times["Rg_Time_MS"],
            ],
            labels = ["Rg_Time_P", "Rg_Time_S", "Rg_Time_T",
                      "Rg_Time_IN", "Rg_Time_SP", "Rg_Time_PU", "Rg_Time_MS"])
plt.ylabel("Seconds")
plt.title("Range Time Distribution (seconds)")
plt.show()


#%%



