# Imports.
import streamlit as st 
import pandas as pd
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style("dark")
#---------------------------------------------------------------------------------------------------------------------------

# Helper Functions.
# Load Data from external source.
@st.cache   # Storing the cache value to reduce time in downloading for further use.
def load_data():
	df = pd.read_csv("https://raw.githubusercontent.com/ThuwarakeshM/PracticalML-KMeans-Election/master/voters_demo_sample.csv")
	return df

df = load_data()

def run_kmeans(df, n_clusters=2):
	kmeans = KMeans(n_clusters, random_state=0).fit(df[["Age", "Income"]])

	fig, ax = plt.subplots(figsize=(16,9))
	#centroids = kmeans.cluster_centers_

	# Create Scatter Plot.
	ax = sns.scatterplot(
						  ax = ax,
						  x  = df.Age,
						  y  = df.Income,
						  hue= kmeans.labels_,
						  palette = sns.color_palette("colorblind", n_colors=n_clusters),
						  legend = None,  
						)
	return fig

#-----------------------------------------------------------------------------------------------------------------------------------

# Main App.
# Create a tile for your app.
st.title("Interactive K-Means Clustering")


# Sidebar.
sidebar = st.sidebar

# Display the dataframe.
df_display = sidebar.checkbox("Display Raw Data", value=True)
if df_display:
    # A description.
    st.write("Here is a dataset used in this analysis:")
    st.write(df)

# Selecting no. of clusters.
n_clusters = sidebar.slider(
							"Select Number of Clusters",
							min_value = 2,
							max_value = 10,	)

# Show cluster scatter plot.
st.write(run_kmeans(df, n_clusters = n_clusters	))

st.write("Now we have **{}** Clusters in this dataset. ".format(n_clusters))