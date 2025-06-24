#!/usr/bin/env python3
"""
Simple test script to verify async functionality of actors.
"""
import asyncio
from actors.actors import vLLMActor, OpenAIActor
from vllm import SamplingParams

async def test_vllm_async():
    """Test async methods with vLLM actor."""
    print("Testing vLLM async methods...")
    
    actor = vLLMActor(
        name="test_async",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        engine_kwargs={
            "gpu_memory_utilization": 0.3,
            "max_model_len": 512,
        },
    )
    
    # Test async generate
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about technology."
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
    )
    
    print("Testing agenerate...")
    async_results = await actor.agenerate(prompts, sampling_params=sampling_params)
    print(f"Async generate completed: {len(async_results)} results")
    for i, result in enumerate(async_results):
        print(f"  Result {i+1}: {result.outputs[0].text[:50]}...")
    
    # Test async chat
    dialogs = [
        [{"role": "user", "content": "Hello, how are you?"}],
        [{"role": "user", "content": "What's 2+2?"}],
        [{"role": "user", "content": "Tell me a joke."}]
    ]
    
    print("\nTesting achat...")
    async_chat_results = await actor.achat(dialogs, sampling_params=sampling_params)
    print(f"Async chat completed: {len(async_chat_results)} results")
    for i, result in enumerate(async_chat_results):
        print(f"  Chat result {i+1}: {result.outputs[0].text[:50]}...")
    
    # Compare with sync versions
    print("\nComparing with sync versions...")
    sync_results = actor.generate(prompts, sampling_params=sampling_params)
    sync_chat_results = actor.chat(dialogs, sampling_params=sampling_params)
    
    print(f"Sync generate: {len(sync_results)} results")
    print(f"Sync chat: {len(sync_chat_results)} results")
    
    print("vLLM async test completed successfully!")

async def test_openai_async():
    """Test async methods with OpenAI actor (mock test)."""
    print("\nTesting OpenAI async methods (will fail without API key)...")
    
    try:
        actor = OpenAIActor(
            name="gpt-3.5-turbo",
            api_key="fake-key-for-testing",
            base_url="https://api.openai.com/v1"
        )
        
        # This will fail but demonstrates the interface
        prompts = ["Hello world"]
        dialogs = [[{"role": "user", "content": "Hello"}]]
        
        # These calls will fail due to invalid API key, but show the async interface
        print("OpenAI async methods are available:")
        print(f"  agenerate method: {hasattr(actor, 'agenerate')}")
        print(f"  achat method: {hasattr(actor, 'achat')}")
        
    except Exception as e:
        print(f"Expected error (no valid API key): {e}")

async def main():
    """Main test function."""
    print("Starting async actor tests...")
    
    # Test vLLM async functionality
    await test_vllm_async()
    
    # Test OpenAI async interface (without making actual calls)
    await test_openai_async()
    
    print("\nAll async tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
