REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
When you are ready to give the final answer:

When you are ready to give the final answer:
1) Read and follow these format instructions exactly:
{format_instructions}
2) Output ONLY the final answer, no prose.
3) Wrap it between <final_json> and </final_json>.


Begin!

Question: {input}
Thought:{agent_scratchpad}
"""