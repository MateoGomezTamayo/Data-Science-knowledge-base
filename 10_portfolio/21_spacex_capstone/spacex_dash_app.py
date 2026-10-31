import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")
low, high = df["Payload Mass (kg)"].min(), df["Payload Mass (kg)"].max()
sites = sorted(df["Launch Site"].unique())

app = Dash(__name__)
app.layout = html.Div(style={"fontFamily": "Arial", "maxWidth": "1100px", "margin": "auto"}, children=[
    html.H1("SpaceX Launch Records Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(id="site-dropdown", value="ALL", searchable=True, placeholder="Select a launch site",
                 options=[{"label": "All Sites", "value": "ALL"}] + [{"label": s, "value": s} for s in sites]),
    dcc.Graph(id="success-pie-chart"),
    html.P("Payload range (kg):"),
    dcc.RangeSlider(id="payload-slider", min=0, max=10000, step=1000, value=[low, high],
                    marks={i: str(i) for i in range(0, 10001, 2500)}),
    dcc.Graph(id="success-payload-scatter-chart"),
])


@app.callback(Output("success-pie-chart", "figure"), Input("site-dropdown", "value"))
def pie(site):
    if site == "ALL":
        data = df[df["class"] == 1].groupby("Launch Site").size().reset_index(name="Successes")
        return px.pie(data, names="Launch Site", values="Successes", title="Share of successful landings by site")
    data = df[df["Launch Site"] == site]["class"].map({1: "Success", 0: "Failure"}).value_counts().reset_index()
    data.columns = ["Outcome", "Launches"]
    return px.pie(data, names="Outcome", values="Launches", title=f"Landing outcomes for {site}",
                  color="Outcome", color_discrete_map={"Success": "#1F7A6D", "Failure": "#C8452F"})


@app.callback(Output("success-payload-scatter-chart", "figure"),
              [Input("site-dropdown", "value"), Input("payload-slider", "value")])
def scatter(site, payload):
    data = df[df["Payload Mass (kg)"].between(payload[0], payload[1])]
    if site != "ALL":
        data = data[data["Launch Site"] == site]
    return px.scatter(data, x="Payload Mass (kg)", y="class", color="Booster Version Category",
                      title="Payload vs. landing outcome")


if __name__ == "__main__":
    app.run(port=8050, debug=False)
