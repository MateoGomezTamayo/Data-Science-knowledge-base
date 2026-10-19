"""
Final Assignment: Part 2 - Create Dashboard with Plotly and Dash
Automobile Sales Statistics Dashboard

Run:  python Automobile_Sales_Statistics_Dashboard.py
Then open the printed URL (e.g. http://127.0.0.1:8050) in a browser.
"""

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the data
data = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/"
    "historical_automobile_sales.csv"
)

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Automobile Statistics Dashboard"

# List of years
year_list = [i for i in range(1980, 2024, 1)]

# ---------------------------------------------------------------------------
# TASK 2.1: Create the dropdown menus and the overall application layout
# ---------------------------------------------------------------------------
dropdown_options = [
    {"label": "Yearly Statistics", "value": "Yearly Statistics"},
    {"label": "Recession Period Statistics", "value": "Recession Period Statistics"},
]

app.layout = html.Div(
    [
        # TASK 2.1: Title
        html.H1(
            "Automobile Sales Statistics Dashboard",
            style={"textAlign": "center", "color": "#503D36", "font-size": 24},
        ),
        # TASK 2.2: Add two dropdown menus
        html.Div(
            [
                html.Label("Select Statistics:"),
                dcc.Dropdown(
                    id="dropdown-statistics",
                    options=dropdown_options,
                    value="Select Statistics",
                    placeholder="Select a report type",
                    style={
                        "width": "80%",
                        "padding": "3px",
                        "font-size": "20px",
                        "text-align-last": "center",
                    },
                ),
            ]
        ),
        html.Div(
            dcc.Dropdown(
                id="select-year",
                options=[{"label": i, "value": i} for i in year_list],
                value="Select Year",
                placeholder="Select a year",
                style={
                    "width": "80%",
                    "padding": "3px",
                    "font-size": "20px",
                    "text-align-last": "center",
                },
            )
        ),
        # TASK 2.3: Add a division for output display
        html.Div(
            [
                html.Div(
                    id="output-container",
                    className="chart-grid",
                    style={"display": "flex"},
                ),
            ]
        ),
    ]
)


# ---------------------------------------------------------------------------
# TASK 2.4: Creating Callbacks
# Enable/disable the year dropdown based on the selected statistics
# ---------------------------------------------------------------------------
@app.callback(
    Output(component_id="select-year", component_property="disabled"),
    Input(component_id="dropdown-statistics", component_property="value"),
)
def update_input_container(selected_statistics):
    if selected_statistics == "Yearly Statistics":
        return False
    else:
        return True


# Callback for plotting - render the appropriate report
@app.callback(
    Output(component_id="output-container", component_property="children"),
    [
        Input(component_id="dropdown-statistics", component_property="value"),
        Input(component_id="select-year", component_property="value"),
    ],
)
def update_output_container(selected_statistics, input_year):
    # -------------------------------------------------------------------
    # TASK 2.5: Create and display graphs for Recession Report Statistics
    # -------------------------------------------------------------------
    if selected_statistics == "Recession Period Statistics":
        recession_data = data[data["Recession"] == 1]

        # Plot 1: Automobile sales fluctuation over Recession Period (year wise)
        yearly_rec = (
            recession_data.groupby("Year")["Automobile_Sales"].mean().reset_index()
        )
        R_chart1 = dcc.Graph(
            figure=px.line(
                yearly_rec,
                x="Year",
                y="Automobile_Sales",
                title="Average Automobile Sales fluctuation over Recession Period",
            )
        )

        # Plot 2: Average number of vehicles sold by vehicle type
        average_sales = (
            recession_data.groupby("Vehicle_Type")["Automobile_Sales"]
            .mean()
            .reset_index()
        )
        R_chart2 = dcc.Graph(
            figure=px.bar(
                average_sales,
                x="Vehicle_Type",
                y="Automobile_Sales",
                title="Average Number of Vehicles Sold by Vehicle Type",
            )
        )

        # Plot 3: Total expenditure share by vehicle type during recessions
        exp_rec = (
            recession_data.groupby("Vehicle_Type")["Advertising_Expenditure"]
            .sum()
            .reset_index()
        )
        R_chart3 = dcc.Graph(
            figure=px.pie(
                exp_rec,
                values="Advertising_Expenditure",
                names="Vehicle_Type",
                title="Total Advertising Expenditure Share by Vehicle Type during Recessions",
            )
        )

        # Plot 4: Effect of unemployment rate on vehicle type and sales
        unemp_data = (
            recession_data.groupby(["unemployment_rate", "Vehicle_Type"])[
                "Automobile_Sales"
            ]
            .mean()
            .reset_index()
        )
        R_chart4 = dcc.Graph(
            figure=px.bar(
                unemp_data,
                x="unemployment_rate",
                y="Automobile_Sales",
                color="Vehicle_Type",
                labels={
                    "unemployment_rate": "Unemployment Rate",
                    "Automobile_Sales": "Average Automobile Sales",
                },
                title="Effect of Unemployment Rate on Vehicle Type and Sales",
            )
        )

        return [
            html.Div(
                className="chart-item",
                children=[html.Div(children=R_chart1), html.Div(children=R_chart2)],
                style={"display": "flex"},
            ),
            html.Div(
                className="chart-item",
                children=[html.Div(children=R_chart3), html.Div(children=R_chart4)],
                style={"display": "flex"},
            ),
        ]

    # -------------------------------------------------------------------
    # TASK 2.6: Create and display graphs for Yearly Report Statistics
    # -------------------------------------------------------------------
    elif (input_year and input_year != "Select Year") and selected_statistics == "Yearly Statistics":
        yearly_data = data[data["Year"] == input_year]

        # Plot 1: Yearly Automobile sales using line chart for the whole period
        yas = data.groupby("Year")["Automobile_Sales"].mean().reset_index()
        Y_chart1 = dcc.Graph(
            figure=px.line(
                yas,
                x="Year",
                y="Automobile_Sales",
                title="Yearly Average Automobile Sales",
            )
        )

        # Plot 2: Total Monthly Automobile sales using line chart
        mas = yearly_data.groupby("Month")["Automobile_Sales"].sum().reset_index()
        Y_chart2 = dcc.Graph(
            figure=px.line(
                mas,
                x="Month",
                y="Automobile_Sales",
                title="Total Monthly Automobile Sales",
            )
        )

        # Plot 3: Average vehicles sold by vehicle type in the selected year (bar chart)
        avr_vdata = (
            yearly_data.groupby("Vehicle_Type")["Automobile_Sales"]
            .mean()
            .reset_index()
        )
        Y_chart3 = dcc.Graph(
            figure=px.bar(
                avr_vdata,
                x="Vehicle_Type",
                y="Automobile_Sales",
                title=f"Average Vehicles Sold by Vehicle Type in the year {input_year}",
            )
        )

        # Plot 4: Total Advertisement Expenditure for each vehicle (pie chart)
        exp_data = (
            yearly_data.groupby("Vehicle_Type")["Advertising_Expenditure"]
            .sum()
            .reset_index()
        )
        Y_chart4 = dcc.Graph(
            figure=px.pie(
                exp_data,
                values="Advertising_Expenditure",
                names="Vehicle_Type",
                title="Total Advertisement Expenditure for Each Vehicle",
            )
        )

        return [
            html.Div(
                className="chart-item",
                children=[html.Div(children=Y_chart1), html.Div(children=Y_chart2)],
                style={"display": "flex"},
            ),
            html.Div(
                className="chart-item",
                children=[html.Div(children=Y_chart3), html.Div(children=Y_chart4)],
                style={"display": "flex"},
            ),
        ]

    else:
        return None


# Run the Dash app
if __name__ == "__main__":
    app.run(debug=False)
