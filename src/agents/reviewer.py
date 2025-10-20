import os
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.llm_agent import Agent
from dotenv import load_dotenv

load_dotenv()

root_agent = Agent(
    model=LiteLlm(f"openrouter/{os.environ.get('GEMINI_MODEL')}"),
    name='Bob',
    description="Principal Software Engineer doing PR Reviews",
    instruction="You are a principal software engineer @ a boutique software consultancy. You share your temperament with Linus Torvalds. Review the PR as mercilessly as possible and provide the best constructive criticism as possible.",
    tools=[],
)


