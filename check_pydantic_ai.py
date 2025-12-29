import asyncio
import logging

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class M(BaseModel):
    """Sample output model for the PydanticAI test."""

    x: int


agent = Agent("openai:gpt-4o", output_type=M)


async def main() -> None:
    """Run a simple PydanticAI test invocation."""
    res = await agent.run("test", model=TestModel(custom_output_args={"x": 1}))
    logging.info("Output: %s", res.output)
    logging.info("Type of Output: %s", type(res.output))


asyncio.run(main())
