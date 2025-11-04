import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# Netflix Theme Colors
NETFLIX_COLORS = ["#E50914", "#B20710", "#221F1F", "#FFFFFF", "#737373"]

def netflix_style(fig):
    # Applies Netflix-style theming to Plotly charts
    fig.update_traces(marker_color="#E50914", selector=dict(type="bar"))
    fig.update_layout(
        template=template,
        paper_bgcolor="#0e1117" if theme_choice == "Dark" else "#ffffff",
        plot_bgcolor="#0e1117" if theme_choice == "Dark" else "#ffffff",
        font=dict(color="white" if theme_choice == "Dark" else "black"),
        title_font=dict(size=20, color="#E50914", family="Arial Black"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="white" if theme_choice == "Dark" else "black")
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# Getting the current directory (where this script is)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Building the path to the dataset file
file_path = os.path.join(current_dir, "cleaned_netflix.csv")

# Reading the CSV safely
df = pd.read_csv(file_path)

# Navbar title
st.sidebar.title("Navigation")

# Dark Theme Toggle
theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"])
template = "plotly_dark" if theme_choice == "Dark" else "plotly_white"

# Dynamic Streamlit Theme
if theme_choice == "Dark":
    st.markdown("""
        <style>
        /* App + sidebar */
        .stApp { background-color: #0e1117; color: #fafafa; }
        section[data-testid="stSidebar"] { background-color: #1e1e1e; }
        section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

        /* Fix top navigation bar for dark mode */
        header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: white !important;
        }
        header[data-testid="stHeader"] * {
        color: white !important;
        }

        /* Labels and text */
        h1, h2, h3, h4, h5, h6, label, .css-16huue1, .css-10trblm {
            color: #fafafa !important;
        }

        /* Metrics */
        div[data-testid="stMetricValue"] { color: #00c4ff !important; }

        /* Buttons */
        div.stButton > button:first-child {
            background-color: #333 !important;
            color: white !important;
            border: 1px solid #888 !important;
            border-radius: 8px !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #444 !important;
            color: #ff4b4b !important;
        }

        /* Download CSV button */
        div[data-testid="stDownloadButton"] > button {
            background-color: #222 !important;
            color: #fafafa !important;
            border: 1px solid #666 !important;
            border-radius: 8px !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #444 !important;
            border-color: #ff4b4b !important;
            color: #ff4b4b !important;
        }

        /* Fix File Uploader + Browse Files Button */
        .stFileUploader label {
            color: #fafafa !important;
            font-weight: 500 !important;
        }
        /* Uploader box */
        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
            background-color: #1e1e1e !important;
            border: 1px dashed #666 !important;
            border-radius: 10px !important;
            color: #ccc !important;
        }
        .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #ff4b4b !important;
            color: #ff4b4b !important;
        }
        /* Browse Files button itself */
        .stFileUploader input[type="file"]::file-selector-button {
            background-color: #333 !important;
            color: #fafafa !important;
            border: 1px solid #777 !important;
            border-radius: 6px !important;
            padding: 0.3em 1em !important;
            cursor: pointer !important;
        }
        .stFileUploader input[type="file"]::file-selector-button:hover {
            background-color: #ff4b4b !important;
            color: white !important;
            border-color: #ff4b4b !important;
        }

        /* Input + Multiselects */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiselect > div > div {
            color: white !important;
            background-color: #262730 !important;
        }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #000000; }
        section[data-testid="stSidebar"] { background-color: #f0f2f6; }
        section[data-testid="stSidebar"] * { color: #000000 !important; }

        h1, h2, h3, h4, h5, h6, label, .css-16huue1, .css-10trblm {
            color: #000000 !important;
        }

        div[data-testid="stMetricValue"] { color: #ff4b4b !important; }

        /* Buttons */
        div.stButton > button:first-child {
            background-color: #f5f5f5 !important;
            color: black !important;
            border: 1px solid #bbb !important;
            border-radius: 8px !important;
        }
        div.stButton > button:first-child:hover {
            background-color: #e6e6e6 !important;
            color: #ff4b4b !important;
        }

        /* Download CSV */
        div[data-testid="stDownloadButton"] > button {
            background-color: #f5f5f5 !important;
            color: #000 !important;
            border: 1px solid #bbb !important;
            border-radius: 8px !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background-color: #e6e6e6 !important;
            color: #ff4b4b !important;
        }

        /* File uploader */
        .stFileUploader label { color: #000000 !important; font-weight: 500 !important; }
        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
            background-color: #fafafa !important;
            border: 1px dashed #bbb !important;
            border-radius: 10px !important;
            color: #000 !important;
        }
        .stFileUploader div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #ff4b4b !important;
        }
        .stFileUploader input[type="file"]::file-selector-button {
            background-color: #f5f5f5 !important;
            color: #000 !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
            padding: 0.3em 1em !important;
            cursor: pointer !important;
        }
        .stFileUploader input[type="file"]::file-selector-button:hover {
            background-color: #ff4b4b !important;
            color: white !important;
            border-color: #ff4b4b !important;
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiselect > div > div {
            color: black !important;
            background-color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)


text_color = "white" if theme_choice == "Dark" else "black"

st.set_page_config(page_title="Netflix Analysis!", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Netflix Movies and TV Shows EDA")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

#Upload Dataset
st.sidebar.header("Upload Netflix Dataset")
base_path = os.path.dirname(__file__)
default_path = os.path.join(base_path, "cleaned_netflix.csv")

uploaded_file = st.sidebar.file_uploader("Upload your Netflix CSV file", type=["csv"])

def is_netflix_csv(df: pd.DataFrame) -> bool:
    """Check if the uploaded CSV looks like a Netflix dataset."""
    netflix_keywords = {"title", "type", "release_year", "country", "listed_in"}
    cols = set(df.columns.str.lower())
    return len(cols.intersection(netflix_keywords)) >= 3  # must match at least 3 key columns

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        if not is_netflix_csv(df):
            st.sidebar.error("This doesn’t appear to be a valid Netflix dataset.")
            st.sidebar.info("Please upload a CSV with columns like title, type, release_year, country, etc.")
            st.stop()
        st.sidebar.success("Netflix dataset uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.stop()

elif os.path.exists(default_path):
    df = pd.read_csv(default_path, encoding="ISO-8859-1")
    st.sidebar.info("Using default cleaned Netflix dataset.")
else:
    st.sidebar.error("No dataset found! Please upload a Netflix CSV file to continue.")
    st.stop()


# Data Validation
required_cols = ["title", "type", "release_year", "country", "duration"]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}. Please upload a valid Netflix dataset.")
    st.stop()

# Dataset Preview
st.title("Netflix Movies & TV Shows Dashboard")
st.markdown("Explore Netflix data interactively — upload your own dataset or use the default one!")

with st.expander("Preview Dataset"):
    st.dataframe(df.head())

st.sidebar.markdown("---")

col1, col2 = st.columns((2))
df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors='coerce')
print(df.columns)

#Getting the starting and ending date
startDate = pd.to_datetime(df["date_added"]).min()
endDate = pd.to_datetime(df["date_added"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))
with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["date_added"] >= date1) & (df["date_added"] <= date2)].copy()

# Sidebar Filters
st.sidebar.header("Filters")

type_filter = st.sidebar.multiselect(
    "Select Type",
    options=df["type"].unique(),
    default=df["type"].unique()
)

genres = df["listed_in"].dropna().str.split(", ").explode().unique()

genre_filter = st.sidebar.multiselect(
    "Select Genre",
    options=genres,
)

df_filtered = df[df["type"].isin(type_filter)]

if genre_filter:
    df_filtered = df_filtered[df_filtered["listed_in"].str.contains("|".join(genre_filter))]

# KPIs (Small Metrics at Top)
col1, col2, col3 = st.columns(3)
col1.metric("Total Titles", df_filtered.shape[0])
col2.metric("Movies", df_filtered[df_filtered["type"]=="Movie"].shape[0])
col3.metric("TV Shows", df_filtered[df_filtered["type"]=="TV Show"].shape[0])

# Visualizations

# Titles Released by Year
df_filtered['release_year'] = df_filtered['release_year'].astype(int)
year_count = df_filtered['release_year'].value_counts().sort_index()

fig_year = px.bar(
    x=year_count.index,
    y=year_count.values,
    labels={"x": "Release Year", "y": "Count"},
    title="Number of Titles per Year",
    color_discrete_sequence=["#E50914"]
)
fig_year = netflix_style(fig_year)
st.plotly_chart(fig_year, use_container_width=True)

# Movies vs TV Shows Pie Chart
fig_pie = px.pie(
    df_filtered,
    names='type',
    title='Distribution by Type',
    color_discrete_sequence=["#E50914", "#B20710"]
)
fig_pie.update_traces(
    textinfo='percent+label',
    pull=[0.05, 0.05],
    hoverinfo='label+percent'
)
fig_pie = netflix_style(fig_pie)
st.plotly_chart(fig_pie, use_container_width=True)

# Top 10 Countries
country = df_filtered['country'].dropna().str.split(", ").explode()
country_count = country.value_counts().head(10)

fig_country = px.bar(
    x=country_count.values,
    y=country_count.index,
    orientation='h',
    labels={"x": "Count", "y": "Country"},
    title="Top 10 Countries with Most Titles",
    color_discrete_sequence=["#E50914"]
)
fig_country = netflix_style(fig_country)
st.plotly_chart(fig_country, use_container_width=True)

# Ratings Distribution
st.subheader("Distribution of Ratings")

rating_count = df_filtered['rating'].value_counts().head(15)
fig_rating = px.bar(
    x=rating_count.index,
    y=rating_count.values,
    labels={'x': 'Rating', 'y': 'Number of Titles'},
    title="Most Common Ratings on Netflix",
    color_discrete_sequence=["#E50914"]
)
fig_rating = netflix_style(fig_rating)
st.plotly_chart(fig_rating, use_container_width=True)

# Top Directors
directors = df_filtered['director'].dropna().str.split(', ').explode()
top_directors = directors.value_counts().head(10)

fig_dir = px.bar(
    x=top_directors.values,
    y=top_directors.index,
    orientation='h',
    labels={'x': 'Count', 'y': 'Director'},
    title="Top 10 Most Frequent Directors",
    color_discrete_sequence=["#B20710"]
)
fig_dir = netflix_style(fig_dir)
st.plotly_chart(fig_dir, use_container_width=True)

# Top Actors
actors = df_filtered['cast'].dropna().str.split(', ').explode()
top_actors = actors.value_counts().head(10)

fig_act = px.bar(
    x=top_actors.values,
    y=top_actors.index,
    orientation='h',
    labels={'x': 'Count', 'y': 'Actor'},
    title="Top 10 Most Frequent Actors",
    color_discrete_sequence=["#E50914"]
)
fig_act = netflix_style(fig_act)
st.plotly_chart(fig_act, use_container_width=True)

# Heatmap (Release Trends)
st.subheader("Release Trends by Country")

heatmap_data = (
    df_filtered.dropna(subset=['country', 'release_year'])
    .assign(country=lambda x: x['country'].str.split(', ').str[0])
    .groupby(['country', 'release_year'])
    .size()
    .reset_index(name='count')
)
top_countries = heatmap_data.groupby('country')['count'].sum().nlargest(10).index
heatmap_data = heatmap_data[heatmap_data['country'].isin(top_countries)]
pivot = heatmap_data.pivot(index='country', columns='release_year', values='count').fillna(0)

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(
    pivot, cmap="Reds", linewidths=0.4, linecolor="black",
    cbar_kws={'label': 'Count'}
)
ax.set_title("Release Trends by Country (Top 10)", fontsize=18, color="#E50914", fontweight="bold")
ax.set_xlabel("Release Year", fontsize=12, color="white" if theme_choice == "Dark" else "black")
ax.set_ylabel("Country", fontsize=12, color="white" if theme_choice == "Dark" else "black")
st.pyplot(fig)

# WordCloud (Titles + Descriptions)
st.subheader("Wordcloud for Titles & Descriptions")

text = " ".join(df_filtered["title"].astype(str)) + " " + " ".join(df_filtered["description"].astype(str))

bg_color = "black" if theme_choice == "Dark" else "white"
colormap = "Reds" if theme_choice == "Dark" else "Blues"

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color=bg_color,
    colormap=colormap,
    max_words=150,
    min_font_size=10,
    collocations=False,
    contour_color="#E50914",
    contour_width=2
).generate(text)

fig_wc, ax_wc = plt.subplots(figsize=(12,6))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
st.pyplot(fig_wc)

# Combine all text
text = " ".join(df_filtered["title"].astype(str)) + " " + " ".join(df_filtered["description"].astype(str))

# Theme-based customization
bg_color = "black" if theme_choice == "Dark" else "white"
colormap = "Set3" if theme_choice == "Dark" else "viridis"

# Generate wordcloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color=bg_color,
    colormap=colormap,
    max_words=150,
    min_font_size=10,
    collocations=False,   # prevents duplicate word pairs
    contour_color='white' if theme_choice == 'Dark' else 'black',
    contour_width=1
).generate(text)

# Display it
fig_wc, ax_wc = plt.subplots(figsize=(12,6))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
st.pyplot(fig_wc)

# Download Button for Filtered Dataset
st.subheader("Download Filtered Dataset")

st.write(f"Currently showing **{df_filtered.shape[0]}** filtered titles.")

csv = df_filtered.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Filtered CSV",
    data=csv,
    file_name="filtered_netflix_data.csv",
    mime="text/csv",
    use_container_width=True
)



