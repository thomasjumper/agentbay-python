"""AgentBay integrations for popular AI frameworks.

Available integrations:

CrewAI (``pip install agentbay[crewai]``)::

    from agentbay.integrations.crewai import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    agent = Agent(role="...", memory=memory)

LangChain - BaseMemory (``pip install agentbay[langchain]``)::

    from agentbay.integrations.langchain import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    chain = ConversationChain(llm=llm, memory=memory)

LangChain - Tool (``pip install agentbay[langchain]``)::

    from agentbay.integrations.langchain import AgentBayMemoryTool
    tool = AgentBayMemoryTool(api_key="ab_live_...", project_id="...")
    agent = initialize_agent(tools=[tool], llm=llm, ...)

AutoGen / AG2 (``pip install agentbay[autogen]``)::

    from agentbay.integrations.autogen import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    memory.attach(assistant_agent)

Install all integrations at once::

    pip install agentbay[all]
"""
