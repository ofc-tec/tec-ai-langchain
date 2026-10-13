import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def bar_plot(
    labels: list[str],
    values: list[float],
    title: str = "Bar plot"
) -> str:
    """Create a bar plot for comparing categorical values."""

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.tight_layout()

    plt.savefig("bar_plot.png")
    ##plt.show()  ### BREAKS AGNET
    plt.close()

    return "Bar plot created."


@tool
def scatter_plot(
    x: list[float],
    y: list[float],
    title: str = "Scatter plot"
) -> str:
    """Create a scatter plot for two numeric variables."""

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.tight_layout()

    plt.savefig("scatter_plot.png")
    ##plt.show()
    plt.close()

    return "Scatter plot created."


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

plot_agent = create_agent(
    model=llm,
    tools=[bar_plot, scatter_plot],
    system_prompt="""
    You are a visualization specialist.

    #Use bar_plot to compare categories.
    Use scatter_plot to show relationships between numeric variables.
    """
)


if __name__ == "__main__":
    result = plot_agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Plot North=120, South=90, East=150, West=110"
        }]
    })

    print(result["messages"][-1].content[0] ["text"])



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

    print(result["messages"][-1].content[0] ["text"])