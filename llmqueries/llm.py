import functools
import hashlib
import json
import os.path
import time
import uuid
from collections import deque
from pathlib import Path

import tiktoken
#import barebonesllmchat
import numpy as np
import openai
from openai import AuthenticationError
from tenacity import retry, stop_after_attempt, wait_exponential

# Define rate limit parameters (adjust based on API tier)
#TOKENS_PER_MINUTE = 10000  # Example TPM limit
TOKENS_PER_MINUTE_DICT = {
    "gpt-3.5-turbo": 30000,
    "gpt-4": 10000,
    "gpt-4-0613": 10000,
    "gpt-3.5-turbo-16k": 30000,
    "gpt-4.1-nano-2025-04-14": 200000,
    "gpt-4.1-mini-2025-04-14": 200000,
    "gpt-4.1-2025-04-14": 10000,
    "gpt-5-2025-08-07": 30000,
"gpt-5-mini": 200000,
    "gpt-5-mini-2025-08-07": 200000,
    "gpt-5-nano-2025-08-07": 200000,
"gpt-5-nano": 200000,
    "gpt-4-turbo-2024-04-09": 10000,
    "bbllm": np.inf
}
MAX_COMPLETION_TOKENS_INSTEAD_OF_MAX_TOKENS = {
    "gpt-5-2025-08-07": True,
}
WINDOW_SIZE = 65  # Time window in seconds for rate limit tracking

token_usage_log = deque()  # Store (timestamp, tokens_used)


def log_retry_attempt(retry_state):
    print(f"Retry attempt {retry_state.attempt_number}")
    if retry_state.outcome.failed:
        if isinstance(retry_state.outcome._exception, KeyError) or isinstance(retry_state.outcome._exception, AuthenticationError):
            raise retry_state.outcome._exception
        print(f"Exception: {retry_state.outcome.exception()}")
    wait_time = retry_state.idle_for if retry_state.idle_for else 0
    print(f"Waiting {wait_time:.2f} seconds before retrying...")


def enforce_rate_limit(gpt_version, requested_tokens):
    global token_usage_log
    current_time = time.time()

    # Remove old token usage data outside the time window
    while token_usage_log and token_usage_log[0][0] < current_time - WINDOW_SIZE:
        token_usage_log.popleft()

    # Calculate total tokens used in the last minute
    tokens_used_last_minute = sum(tokens for _, tokens in token_usage_log)

    # If the next request exceeds the rate limit, wait until tokens refresh
    if tokens_used_last_minute + requested_tokens > TOKENS_PER_MINUTE_DICT[gpt_version.lower()]:
        oldest_request_time = token_usage_log[0][0] if token_usage_log else current_time
        wait_time = (oldest_request_time + WINDOW_SIZE) - current_time
        if wait_time > 0:
            print(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    # Log the current request
    token_usage_log.append((time.time(), requested_tokens))


@retry(wait=wait_exponential(multiplier=1, min=4, max=30), stop=stop_after_attempt(5), before_sleep=log_retry_attempt)
def _LLM_retry(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0):
    enforce_rate_limit(gpt_version, max_tokens)

    if (not isinstance(prompt, dict)) and (not isinstance(prompt, list)):
        prompt = [{"role": "user", "content": prompt}]


    if "bbllm" in gpt_version:
        _, text = barebonesllmchat.terminal.openaispoof.ChatCompletion.create(
            prompt,
            max_new_tokens=max_tokens,
            temperature=np.clip(temperature, 0.1, 1.0)
        )
        ret = _, text.strip()

    elif "gpt" not in gpt_version:
        if MAX_COMPLETION_TOKENS_INSTEAD_OF_MAX_TOKENS.get(gpt_version, False):
            response = openai.Completion.create(
            model=gpt_version,
            prompt=prompt,
            max_completion_tokens=max_tokens,
            temperature=1.0,
            stop=stop,
            logprobs=logprobs,
            frequency_penalty=frequency_penalty
        )
        else:
            response = openai.Completion.create(
                model=gpt_version,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                logprobs=logprobs,
                frequency_penalty=frequency_penalty
            )
        token_usage_log.append((time.time(), response["usage"]["total_tokens"]))
        ret = response, response["choices"][0]["text"].strip()

    else:
        if MAX_COMPLETION_TOKENS_INSTEAD_OF_MAX_TOKENS.get(gpt_version, False):
            response = openai.responses.create(
                model=gpt_version,
                input=prompt,
                #max_completion_tokens=max_tokens,
                #temperature=1.0,
                #frequency_penalty=frequency_penalty
            )
        else:
            response = openai.responses.create(
                model=gpt_version,
                input=prompt,
                #max_tokens=max_tokens,
                #temperature=temperature,
                #frequency_penalty=frequency_penalty
            )
        token_usage_log.append((time.time(), response.usage.total_tokens))
        ret = response, response.output_text.strip()
    return ret

from diskcache import FanoutCache, Cache, Deque
CACHEPATH = "/".join(__file__.split("/")[:-1]) + "/diskcache"
cache = FanoutCache(CACHEPATH, shards=64)

@cache.memoize()
def _LLM_cache(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0, ignore_cache=False):
    print("LLM QUERY: Could not find prompt in cache, querying OpenAI API.")
    ret = _LLM_retry(prompt, gpt_version, max_tokens, temperature, stop, logprobs, frequency_penalty)
    return ret

#from termcolor import colored

def LLM(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0, ignore_cache=False):
    if ignore_cache:
        ignore_cache = uuid.uuid4().hex # adds random input to fool diskcache

    a, response = _LLM_cache(prompt, gpt_version, max_tokens, temperature, stop, logprobs, frequency_penalty, ignore_cache=ignore_cache)

    return a, response

def set_api_key(openai_api_key):
    if os.path.exists(openai_api_key + '.txt'):
        openai.api_key = Path(openai_api_key + '.txt').read_text()
    else:
        openai.api_key = openai_api_key

def approx_num_tokens(llm, text):
    encoding = tiktoken.encoding_for_model(llm)
    tokens = encoding.encode(text)
    return len(tokens)

if __name__ == "__main__":
    print(__file__)