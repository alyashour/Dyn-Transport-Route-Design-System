import pandas as pd
import plotly.express as px

df = pd.read_csv('dataset_generator/viz_ready_timeline.csv')
fig = px.line(df, x='Date', y='Trip_Count', title='Simulated Ridership Over Time')
fig.show()