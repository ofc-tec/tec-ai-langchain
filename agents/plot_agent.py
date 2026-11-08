import matplotlib
matplotlib.use("Agg")

from matplotlib.figure import Figure

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama



@tool
def bar_plot(
    labels: list[str],
    values: list[float],
    title: str = "Bar plot"
) -> str:
    """Create a bar plot for comparing categorical values."""

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    ax.bar(labels, values)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig("bar_plot.png")

    return "Bar plot created: bar_plot.png"


@tool
def scatter_plot(
    x: list[float],
    y: list[float],
    groups: list[str] | None = None,
    title: str = "Scatter plot",
    xlabel: str = "X",
    ylabel: str = "Y",
) -> str:
    """Create a scatter plot for two numeric variables.

    x and y must contain the same number of observations.
    Optionally provide one group label per observation to
    distinguish categories using different colors and markers.
    """

    if len(x) != len(y):
        return f"Error: x has {len(x)} values but y has {len(y)}."

    if groups is not None and len(groups) != len(x):
        return (
            f"Error: groups has {len(groups)} labels "
            f"but x has {len(x)} values."
        )

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    if groups is None:
        ax.scatter(x, y)

    else:
        markers = ["o", "s", "^", "D", "v", "P", "X"]
        categories = list(dict.fromkeys(groups))

        for i, category in enumerate(categories):
            indices = [
                j for j, group in enumerate(groups)
                if group == category
            ]

            ax.scatter(
                [x[j] for j in indices],
                [y[j] for j in indices],
                label=category,
                marker=markers[i % len(markers)],
                alpha=0.7,
            )

        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig("scatter_plot.png")

    return "Scatter plot created: scatter_plot.png"



#llm = ChatGoogleGenerativeAI(
#    model="gemini-3.8-flash",
#    temperature=0,
#)
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
)

plot_agent = create_agent(
    model=llm,
    tools=[bar_plot, scatter_plot],
    system_prompt="""
You are a visualization specialist.

Use bar_plot to compare categories.
Use scatter_plot to show relationships between numeric variables.
"""
)