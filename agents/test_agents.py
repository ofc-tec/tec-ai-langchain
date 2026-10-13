from data_agent import data_agent
from plot_agent import plot_agent


# -------------------------
# Test Data Agent
# -------------------------

result = data_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Inspect data/sales.csv"
        }
    ]
})

print("\nDATA AGENT\n")
print(result["messages"][-1].content)


# -------------------------
# Test Plot Agent
# -------------------------

result = plot_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
            Make a scatter plot with:
            North = 120
            South = 90
            East = 150
            West = 110
            """
        }
    ]
})

print("\nPLOT AGENT\n")
print(result["messages"][-1].content)



# -------------------------
# Test Plot Agent
# -------------------------

result = plot_agent.invoke({
    
    "messages": [
            {
                "role": "user",
                "content": """
                Create a scatter plot showing the relationship between
                advertising spend and sales.

                Advertising spend: [20, 25, 30, 35, 40, 45, 50]
                Sales: [90, 105, 120, 138, 150, 168, 185]

                Title the plot 'Advertising Spend vs Sales'.
                """
            }
        ]
})

print("\nPLOT AGENT\n")
print(result["messages"][-1].content)


