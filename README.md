# 📊 Netflix Movies & TV Shows - EDA Dashboard

An **interactive Streamlit dashboard** for exploring and visualizing insights from Netflix’s movies and TV shows dataset.  
It allows users to upload their own datasets, filter data dynamically, and visualize patterns through charts, heatmaps, and word clouds — all in a sleek Netflix-inspired dark/light theme.

---

## Project Overview

The Netflix EDA Dashboard helps analyze:
- Trends in content releases over the years  
- Distribution between Movies and TV Shows  
- Country-wise content insights  
- Ratings and genre patterns  
- Most frequent directors and actors  
- Keyword importance using WordClouds  

Users can **switch themes**, **filter data**, and **download cleaned CSVs** for further analysis.

---

## Features

- **Dynamic Theme Toggle** (Light / Dark)  
- **File Upload** – Supports CSV, Excel, or TXT datasets  
- **Interactive Filters** – Filter by type, genre, and release date  
- **KPI Metrics** – Displays total titles, movies, and TV shows  
- **Netflix-Styled Visualizations** – Red & black themed charts  
- **Downloadable Data** – Export filtered results as CSV  
- **WordClouds** – Generate keyword clouds from titles and descriptions  

---

## Tech Stack

- **Python 3.12+**
- **Streamlit** – Web app framework  
- **Pandas** – Data manipulation  
- **Plotly** – Interactive charts  
- **Seaborn & Matplotlib** – Statistical and heatmap visuals  
- **WordCloud** – Text visualization  
- **OpenPyXL** – Excel support  

---

## Dataset
The project uses the **Netflix Movies and TV Shows dataset** available on Kaggle
.
It includes details such as:
- Title
- Type (Movie / TV Show)
- Director
- Cast
- Country
- Date Added
- Release Year
- Rating
- Duration
- Genre (Listed In)
- Description

## Visualizations
The dashboard includes:
- Bar Chart → Titles released by year
- Pie Chart → Movies vs TV Shows distribution
- Horizontal Bars → Top countries, directors, and actors
- Heatmap → Release trends by country
- WordClouds → Common words in titles and descriptions

Each chart automatically adapts to the chosen Light/Dark Netflix theme.

## Themes
- Light Theme → Clean white background
- Dark Theme → Netflix-inspired black background with red highlights (#E50914)

Users can toggle themes via the sidebar for a better viewing experience.

## Possible Future Enhancements
- Integration of Machine Learning models to predict content trends.
- Addition of a recommendation system for suggesting similar shows/movies.
- Addition of a time-based slider for analyzing release patterns over decades.
- Including regional sentiment analysis on descriptions.
- Addition of IMDb score integration

## Link : 

