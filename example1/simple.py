#%%
import asyncio 
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt='Be concise, reply with one sentence.',
)

async def get_result(): # or in top-level Jupyter cell:
    result = await agent.run('Where does "hello world" come from?')  
    return result.data

# In IPython/Jupyter:
result = await get_result()  # Direct await in a cell.
print(result)

#%%
print(result.data)
