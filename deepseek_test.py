
from openai import OpenAI

def test_deepseek_openrouter(prompt: str, model: str = "deepseek-chat"):
    """
    Sends a test request to DeepSeek via OpenRouter.
    """
    api_key = "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"
    if not api_key:
        raise ValueError("Please set your OPENROUTER_API_KEY environment variable.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    response = client.chat.completions.create(
        model=f"deepseek/{model}",   # note the "deepseek/" prefix for OpenRouter
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    test_prompt = "Hello from OpenRouter DeepSeek!"
    print("Sending test prompt...")
    try:
        result = test_deepseek_openrouter(test_prompt)
        print("Response from DeepSeek (via OpenRouter):")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
