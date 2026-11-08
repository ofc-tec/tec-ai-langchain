import asyncio

from langsmith import evaluate
from langchain_core.tracers.context import collect_runs
from agents.gcp_data import run_gcp_data


async def run_agent(question):
    with collect_runs() as collector:
        try:
            answer = await run_gcp_data(question)
        except Exception as e:
            answer = f"ERROR: {e}"

    # Extract actual tool executions, in chronological order
    tools = sorted(
        (run for run in collector.traced_runs
         if run.run_type == "tool"),
        key=lambda run: run.start_time,
    )

    trajectory = [
        {
            "tool": run.name,
            "arguments": run.inputs,
            "result": str(run.outputs)[:1000],
            "error": run.error,
        }
        for run in tools
    ]

    return {
        "answer": str(answer),
        "trajectory": trajectory,
    }


def target(inputs):
    return asyncio.run(run_agent(inputs["question"]))


evaluate(
    target,
    data="gcp_data_agent_langsmith_dataset",
    experiment_prefix="gcp-data-agent-trajectory",
)