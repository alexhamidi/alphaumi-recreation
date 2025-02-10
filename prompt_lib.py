prompt_dict = {
"backbone":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You should answer the action of next step in the following format:
The thought to solve the question,
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.
Once you think the question is finished, output conclusion: the final answer of the question or give up if you think you cannot answer this question.
""",

"planner":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
You are the assistant to plan what to do next and whether is caller's or summarizer's turn to answer.
Answer with a following format:
The rationale of the next step, followed by Next: caller or summarizer or give up.""",


"caller":"""You have assess to the following apis:
{doc}
The conversation history is:
{history}
The thought of this step is:
{thought}
Base on the thought make an api call in the following format:
Action: the name of api that should be called in this step, should be exactly in [{tool_names}],
Action Input: the api call request.""",

"summarizer": """Make a conclusion based on the conversation history. Ensure you address the initial user prompt.
{history}""",
}
