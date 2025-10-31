from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama 
from langchain_community.vectorstores import FAISS  # only if you're actually using FAISS
from langchain_core.documents import Document

@tool
def get_text_length(text: str) -> int:
    """Return length of a text by characters."""
    return len(text)

tools = [get_text_length]

#llm = ChatOpenAI(model="gpt-4o-mini")  # or the model you want

#prompt = ChatPromptTemplate.from_messages([
#    ("system", "You are a helpful assistant. Use tools when beneficial."),
#    ("human", "{input}"),
#    MessagesPlaceholder("agent_scratchpad"),
#])

llm = ChatOllama(model="llama3.1:latest", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when beneficial."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])



agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---- optional FAISS demo (requires faiss-cpu installed) ----
# Remove this block if you’re not using FAISS locally.
emb = OpenAIEmbeddings()
docs = [Document(page_content=t) for t in ["hello world", "langchain is neat", "faiss local index"]]
vs = FAISS.from_documents(docs, emb)
# ------------------------------------------------------------

if __name__ == "__main__":
    res = agent_executor.invoke({"input": "Call get_text_length on 'hola'."})
    print(res)
