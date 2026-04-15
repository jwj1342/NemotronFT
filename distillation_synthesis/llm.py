import asyncio
import json
import os
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from prompt import DISTILLATION_COT_SYSTEM_PROMPT
from utils import prompt_format

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


def build_response_payload(response: Any) -> dict[str, Any]:
    choice = response.choices[0] if response.choices else None
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)
    reasoning = ""

    if message is not None:
        for field_name in ("reasoning", "reasoning_content"):
            field_value = getattr(message, field_name, None)
            if isinstance(field_value, str) and field_value:
                reasoning = field_value
                break

    if hasattr(response, "model_dump_json"):
        raw_response = response.model_dump_json(exclude_none=True)
    elif hasattr(response, "model_dump"):
        raw_response = json.dumps(response.model_dump(exclude_none=True), ensure_ascii=False)
    else:
        raw_response = str(response)

    return {
        "reasoning": reasoning,
        "content": getattr(message, "content", "") or "",
        "finish_reason": getattr(choice, "finish_reason", "") or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", "") or "",
        "completion_tokens": getattr(usage, "completion_tokens", "") or "",
        "total_tokens": getattr(usage, "total_tokens", "") or "",
        "raw_response": raw_response,
    }

class BaseLLM:
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    async def _create_completion(self, prompt: str, system_prompt: str, **kwargs):
        return await self.client.chat.completions.create(
            model=self.model,
            temperature=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

    async def get_response_payload(self, prompt: str, system_prompt: str = DISTILLATION_COT_SYSTEM_PROMPT, **kwargs) -> dict[str, Any]:
        async def _call():
            response = await self._create_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs,
            )
            return build_response_payload(response)
        return await retry_on_error(_call)

    async def get_response(
        self,
        prompt: str,
        system_prompt: str = DISTILLATION_COT_SYSTEM_PROMPT,
        **kwargs,
    ) -> tuple[str, str]:
        response_payload = await self.get_response_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )
        return (
            str(response_payload.get("reasoning", "") or ""),
            str(response_payload.get("content", "") or ""),
        )

    async def get_response_batch(
        self,
        queries: list[str],
        concurrency: int = 10,
        **kwargs,
    ) -> list[tuple[str, str]]:
        if concurrency <= 0:
            raise ValueError("concurrency must be greater than 0")

        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(query: str) -> tuple[str, str]:
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
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "step-3.5-flash-2603"):
        self.api_key = api_key if api_key else os.getenv("STEP_API_KEY")
        self.base_url = base_url if base_url else "https://api.stepfun.com/v1"
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _create_completion(self, prompt: str, system_prompt: str, **kwargs):
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="low",
            **kwargs
        )

class ArkPlanLLM(BaseLLM):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "ark-code-latest"):
        self.api_key = api_key if api_key else os.getenv("ARKPLAN_API_KEY")
        self.base_url = base_url if base_url else "https://ark.cn-beijing.volces.com/api/coding/v3"
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

if __name__ == "__main__":
    test_query = """
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
34/44 = 1
41/32 = 9
34|25 = 69
87\64 = 8853
Now, determine the result for: 69/52
"""
    test_answer = "17/"


    import asyncio

    async def test():
        llm = StepLLM()

        if not llm.api_key:
            print("未检测到 STEP_API_KEY，跳过 batch 测试。")
            return

        queries = [
            prompt_format(replacements={"question": test_query, "gold_answer": test_answer})
        ]

        try:
            print("开始测试 batch 请求...")
            responses = await llm.get_response_batch(
                queries,
                concurrency=2,
                max_tokens=254000,
                temperature=0.1,
            )

            print("Batch 测试结果:")
            for idx, (query, (reasoning, content)) in enumerate(zip(queries, responses), start=1):
                print(f"{idx}. query: {query}")
                print(f"\n\n#####reasoning: {reasoning}")
                print(f"\n\n#####content: {content}")
        except Exception as e:
            print(f"batch 测试失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        


    asyncio.run(test())
