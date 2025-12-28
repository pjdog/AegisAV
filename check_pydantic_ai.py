import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class M(BaseModel):
    x: int


agent = Agent("openai:gpt-4o", output_type=M)


async def main():
    res = await agent.run("test", model=TestModel(custom_output_args={"x": 1}))
    print(f"Output: {res.output}")
    print(f"Type of Output: {type(res.output)}")


asyncio.run(main())
