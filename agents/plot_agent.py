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
    title: str = "Scatter plot"
) -> str:
    """Create a scatter plot for two numeric variables."""

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    ax.scatter(x, y)
    ax.set_title(title)

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