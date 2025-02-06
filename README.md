# pydantic_ai_examples
It is a repo for examples in building ai agents using pydantic ai framework
Test

PydanticAI is a robust Python framework designed to streamline the development of production-ready AI agents. It leverages the power and familiarity of Pydantic, a popular data validation and parsing library, to bring type safety, structure, and ease of use to the world of AI agent creation. This tutorial provides a comprehensive, hands-on guide to building your own sophisticated AI agents using PydanticAI.

---

## Page 1: Introduction to PydanticAI and the World of AI Agents

AI agents represent a significant advancement in artificial intelligence, moving beyond simple question-answering towards autonomous programs capable of interacting with their environment, making informed decisions, and achieving complex objectives. PydanticAI offers a structured, principled approach to building such agents. It empowers developers to define agent behavior, integrate with external tools, and process data efficiently, all while maintaining the rigor of Pydantic's validation capabilities.

Key features of PydanticAI that we'll explore in this tutorial include:

* **Model Agnosticism:**  Seamlessly switch between various leading language models (LLMs) like OpenAI, Gemini, and Groq, without significant code changes.
* **Type Safety:**  Harness the power of Pydantic's type hints to ensure data validation and consistency, minimizing runtime errors and improving code reliability.
* **Dependency Injection:** Manage dependencies and configurations gracefully using a built-in dependency injection system, promoting modularity and testability.
* **Tools and Agent Composition:** Extend agent capabilities by integrating external tools and crafting flexible logic flows, enabling your agents to interact with the world around them.
* **Structured Outputs & Validation:** Define the precise structure of your agent's responses using Pydantic models, enabling powerful validation and streamlined data processing.
* **Streaming Responses:** Efficiently handle and process large language model outputs in real-time, providing an enhanced user experience and reducing latency.
* **Robust Error Handling and Retries:** Implement mechanisms for dealing with unexpected errors and retrying operations, increasing the resilience of your agents.
* **Logging and Monitoring:** Integrate with logging and monitoring tools like Pydantic Logfire to gain insights into agent behavior and performance, aiding in debugging and optimization.

This tutorial assumes a basic understanding of Python and the Pydantic library. Let's embark on this journey to create intelligent, robust, and efficient AI agents!

---

## Page 2: Setting Up Your Development Environment

Before we begin building agents, let's set up our development environment. This involves installing PydanticAI, configuring the chosen LLM, and setting up any necessary dependencies.

1. **Installing PydanticAI:** Start by installing the core PydanticAI package using pip:

   ```bash
   pip install pydantic-ai
   ```

2. **LLM-Specific Dependencies:** PydanticAI supports multiple LLMs. You'll need to install the optional dependencies corresponding to your chosen LLM. For example, to use OpenAI models:

   ```bash
   pip install "pydantic-ai[openai]"
   ```

   For other LLMs (e.g., Gemini, Groq), refer to the PydanticAI documentation for specific installation instructions.

3. **API Keys and Authentication:** Obtain the necessary API keys from your LLM provider. For OpenAI, you can get an API key from [platform.openai.com](platform.openai.com). Once you have your key, set it as an environment variable:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

   Refer to the documentation for other LLM providers to learn how to set up authentication.

---

## Page 3: Creating Your First Simple Agent

Let's start by creating a simple agent that can answer basic questions.

```python
from pydantic_ai import Agent

# Initialize the agent with a specified LLM.
agent = Agent("openai:gpt-3.5-turbo") # Or another supported model

# Define a user prompt.
user_prompt = "What is the capital of France?"

# Run the agent synchronously.
result = agent.run_sync(user_prompt)

# Print the agent's response.
print(result.data) # Output: Paris
```

This code initializes an agent using the OpenAI GPT-3.5 model. The `run_sync` method executes the agent synchronously and returns a `RunResult` object containing the agent's response, accessible via `result.data`.

---

## Page 4: Empowering Your Agent with Tools

Tools are the key to extending your agent's functionality beyond basic text generation. They allow your agent to interact with external APIs, databases, or other services to retrieve information or perform actions.

```python
from pydantic_ai import Agent, RunContext, tool
from datetime import date

# Define a tool to get the current date.
@tool
async def get_date(ctx: RunContext) -> str:  
  """Returns the current date."""
  today = date.today()
  return str(today)

# Initialize the agent with the tool.
agent = Agent("openai:gpt-3.5-turbo", tools=[get_date])

# Run the agent with a prompt that requires the tool.
result = agent.run_sync("What is today's date?")

# Print the agent's response.
print(result.data) #  Will include the current date.
```

Here, the `get_date` function is decorated with `@tool`. This registers it as a tool that the agent can use. When the agent encounters a prompt requiring the current date, it automatically calls this tool to retrieve the information and incorporates it into its response.

---

## Page 5:  Structured Outputs, Validation, and Enhanced Responses

PydanticAI empowers you to define structured output formats and leverage Pydantic's validation capabilities for enhanced responses.

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class ProductInfo(BaseModel):
  name: str = Field(..., description="Name of the product.")
  price: float = Field(..., gt=0, description="Price of the product.")
  category: str

# Define a tool function to get fake product info (replace with actual API call later).
@tool
def get_product_info(name: str):
  # Replace this with a call to your product API
  return {"name": name, "price": 25.99, "category": "Electronics"}



agent = Agent("openai:gpt-3.5-turbo", tools=[get_product_info], result_type=ProductInfo)

result = agent.run_sync("Get me info on the latest iPhone.")

if isinstance(result.data, ProductInfo):  # Handle cases where the LLM didn't return structured data correctly
    print(f"Product: {result.data.name}")
    print(f"Price: ${result.data.price}")
    print(f"Category: {result.data.category}")
else:
    print("The LLM did not return valid product information.")

```

This utilizes a `ProductInfo` model to define the expected output structure. PydanticAI handles data validation, ensuring the agent's response adheres to the specified types and constraints.

---



## Page 6: Dynamic Prompts and Contextual Conversations

Dynamic system prompts enable you to create more engaging and contextual conversations by tailoring the agent's instructions based on runtime information.

```python
from pydantic_ai import Agent, RunContext

agent = Agent(deps_type=str)  # We'll use a string for dependencies

@agent.system_prompt
async def dynamic_prompt(ctx: RunContext[str]) -> str:
  return f"You are now talking to user: {ctx.deps}"

result = agent.run_sync("Hello there!", deps="Bob")  # 'Bob' will be injected into the system prompt.
print(result.data)



```

The `dynamic_prompt` function is used to generate the system prompt based on the dependency, which in this example is the user's name.

---



## Page 7: Streaming Responses for Real-time Interactions

Streaming responses provide a powerful way to handle large language model output, allowing you to process and display information in real-time, enhancing user experience and minimizing latency.

```python
from pydantic_ai import Agent
import asyncio

agent = Agent("openai:gpt-3.5-turbo") 

async def main():
    async with agent.run_stream("Tell me a story.") as result:
        async for text in result.stream_text(debounce_by=0.1): # debounce to group text chunks
            print(text, end="", flush=True) # Print immediately without newlines

asyncio.run(main())

```



`agent.run_stream` creates a streamed run, and `result.stream_text` provides an asynchronous iterator over the text stream, allowing you to process chunks of text as they are received.

---



## Page 8: Handling Errors Gracefully: Model Retries and Exceptions

Unexpected errors can occur during agent runs, such as exceeding retry limits or encountering API issues. PydanticAI provides mechanisms to gracefully handle such scenarios.

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

agent = Agent("openai:gpt-3.5-turbo", retries=3)  # Set a retry count

@agent.tool
def my_tool(x: int) -> int:
  if x > 10:
      raise ModelRetry("x must be less than or equal to 10.")
  return x * 2

try:
    result = agent.run_sync("Process x = 12.")

except UnexpectedModelBehavior as e:
  print(f"Error: {e}")
  print(f"Messages: {agent.last_run_messages}")  # Inspect messages for debugging

else:
  print(f"Result: {result.data}") 

```

Setting `retries` in the agent defines the maximum number of retry attempts. If `ModelRetry` is raised within a tool, the agent will attempt to retry the operation up to the specified limit. If the limit is exceeded, `UnexpectedModelBehavior` will be raised, allowing you to handle the error and inspect the conversation history for debugging.



---

## Page 9: Logging and Monitoring with Logfire (Optional)

Integrate Pydantic Logfire to gain detailed insights into your agent's behavior and performance.

```python
from pydantic_ai import Agent
import logfire

logfire.configure() # Configure Logfire

agent = Agent("openai:gpt-3.5-turbo")

result = agent.run_sync("What's the meaning of life?")

# Logfire automatically captures information about the agent run
# for debugging and monitoring.
```

Refer to the Logfire documentation for setup instructions.

---




## Page 10: Advanced Topics and Further Exploration

This extended tutorial provided a solid foundation for building AI agents with PydanticAI. However, the framework offers further advanced capabilities that warrant exploration:

* **Agent Composition:**  Combine multiple agents to create complex workflows and decision-making processes.

* **Custom Models:** Design your own Pydantic models to represent specific data structures and responses.

* **Asynchronous Operations:** Optimize agent performance by utilizing asynchronous tool calls and operations.

* **Unit Testing and Evaluation:**  Learn how to effectively test your agents using testing frameworks and evaluation strategies.

* **Integration with External Libraries:** Connect your agents to a wide range of services and databases for richer interactions.

Delve deeper into the [PydanticAI documentation](https://ai.pydantic.dev/) to explore these advanced features, best practices, and more comprehensive examples. Unlock the full potential of PydanticAI and elevate your AI agent development to the next level!



