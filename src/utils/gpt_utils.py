import sys
import time
from typing import Dict, List

import openai
import requests


def print_error(ex: Exception) -> None:
    print("{0}: {1}".format(ex.__class__.__name__, ex), file=sys.stderr)


def chatgpt_single_turn_inference(
    messages: List[Dict],
    model: str,
    max_tokens: int,
    url: str = None,
    num_return: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.0,
    seed: int = None,
    stop=None,
    timeout: int = 10,
    sleep: int = 1,
):
    """
    messages - a list of messages in a single conversation history
    url - the url of the OpenAI API, if not specified, the default is used
    """

    completions = {"choices": []}

    request_data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": num_return,
        "stop": stop,
        "seed": seed,
    }
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    if url is None:
        url = "https://api.openai.com/v1/chat/completions"

    for i in range(1, 1001):
        try:
            response = requests.post(
                url,
                json=request_data,
                headers=request_headers,
                timeout=timeout,
            )
            response.raise_for_status()  # Check for any errors in the response
            completions = response.json()
            break
        except Exception as e:
            print_error(e)
            # if the error code is 400 (bad request), there is something wrong with the request itself
            if e.response is not None and e.response.status_code == 400:
                print(e.response.json()["error"])
                raise e
            print(f"tried {i} times, sleep for {sleep} seconds ...")
            time.sleep(sleep)

    outputs = [c["message"]["content"] for c in completions["choices"]]
    return outputs

# write a simple test to verify the function
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the capital of California?"},
# ]

# model = "gpt-3.5-turbo"
# max_tokens = 100
# openai.api_key = "YOUR_API_KEY"
# url = YOUR_URL
# outputs = chatgpt_single_turn_inference(messages, model, max_tokens, url=url)
# print(outputs)
