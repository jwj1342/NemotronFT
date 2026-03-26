from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
import os
import asyncio
from prompt import SYSTEM_PROMPT

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

async def retry_on_error(func, max_retries=3, delay=1.0):
    """Simple retry wrapper for async functions"""
    for attempt in range(max_retries):
        try:
            return await func()
        except (APIError, APITimeoutError, RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay * (attempt + 1))

class BaseLLM:
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    async def get_response(self, prompt: str, **kwargs) -> str:
        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=1.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
            **kwargs)
            return response.choices[0].message.content
        return await retry_on_error(_call)

    async def get_response_stream(self, prompt: str, **kwargs):
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=1.0,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
        **kwargs
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def get_response_batch(self, queries: list[str], concurrency: int = 10, **kwargs) -> list[str]:
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")

        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(query: str) -> str:
            async with semaphore:
                return await self.get_response(query, **kwargs)

        tasks = [_worker(query) for query in queries]
        return await asyncio.gather(*tasks)

    
class SiliconflowLLM(BaseLLM):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "Pro/MiniMaxAI/MiniMax-M2.5"):
        self.api_key = api_key if api_key else os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url if base_url else "https://api.siliconflow.cn/v1"
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)


class StepLLM(BaseLLM):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "step-3.5-flash"):
        self.api_key = api_key if api_key else os.getenv("STEP_API_KEY")
        self.base_url = base_url if base_url else "https://api.stepfun.com/v1"
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    async def get_response(self, prompt: str, **kwargs) -> str:
        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=1.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
            **kwargs)
            message = response.choices[0].message
            output_format = "<think>\n{resaoning}\n</think>\n{content}"
            return output_format.format(
                resaoning=getattr(message, "reasoning", "") or "",
                content=message.content or "",
            )
        return await retry_on_error(_call)

class ArkPlanLLM(BaseLLM):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "ark-code-latest"):
        self.api_key = api_key if api_key else os.getenv("ARKPLAN_API_KEY")
        self.base_url = base_url if base_url else "https://ark.cn-beijing.volces.com/api/coding/v3"
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

if __name__ == "__main__":
    test_prompt = """
In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:
96 -> XCVI
45 -> XLV
74 -> LXXIV
39 -> XXXIX
Now, write the number 14 in the Wonderland numeral system.
"""


    import asyncio

    async def test():
        llm = StepLLM()

        if not llm.api_key:
            print("未检测到 STEP_API_KEY，跳过 batch 测试。")
            return

        queries = [
            "1+1 等于几？只回答数字。",
            "把 hello 翻译成中文，只回答翻译结果。",
            test_prompt
        ]

        try:
            print("开始测试 batch 请求...")
            responses = await llm.get_response_batch(
                queries,
                concurrency=2,
                max_tokens=8196,
            )

            print("Batch 测试结果:")
            for idx, (query, response) in enumerate(zip(queries, responses), start=1):
                print(f"{idx}. query: {query}")
                print(f"   response: {response}")
        except Exception as e:
            print(f"batch 测试失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        


    asyncio.run(test())
