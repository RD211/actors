import random

import pytest
import torch
from vllm import SamplingParams

from actors import vLLMActor
from actors.environments import CollaborativeActorConfig, CollaborativeEnvironment
from actors.environments.types import EnvironmentOutput
from actors.rewards.base_conversation_reward import conversation_reward_function


def _merge_actor_outputs(eo: EnvironmentOutput, actor_name):
    merged = None
    for pb in eo.outputs:
        if actor_name in pb:
            for group_name in pb[actor_name]:
                ao = pb[actor_name][group_name]
                if merged is None:
                    merged = ao
                else:
                    merged += ao
    return merged


@pytest.fixture(scope="session", autouse=True)
def _set_seed():
    random.seed(1234)


def _create_actor(name, model_path, engine_kwargs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for vLLM test")
    actor = vLLMActor(name=name, model_path=model_path, engine_kwargs=engine_kwargs)
    yield actor
    actor.kill()


@pytest.fixture(scope="session")
def smollm_actor():
    yield from _create_actor(
        name="smol",
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        engine_kwargs={"gpu_memory_utilization": 0.2},
    )


@pytest.fixture(scope="session")
def qwen_actor():
    yield from _create_actor(
        name="qwen",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        engine_kwargs={"gpu_memory_utilization": 0.2},
    )


@pytest.fixture(scope="session")
def trump_actor():
    yield from _create_actor(
        name="Trump",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        engine_kwargs={"gpu_memory_utilization": 0.2},
    )


@pytest.fixture(scope="session")
def musk_actor():
    yield from _create_actor(
        name="Musk",
        model_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        engine_kwargs={"gpu_memory_utilization": 0.2},
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_one_actor_without_masking(smollm_actor):
    tok = smollm_actor.tokenizer
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=smollm_actor,
                system_prompt="You are SmolLM. Answer this question: ",
                sampling_params=SamplingParams(temperature=0.8),
            )
        ],
        round_spec="smol*3",
        reward_functions=[],
        run_concurrently=False,
        mask_other_agents_for_loss=False,
    )
    batch = {"text": ["Explain overfitting in ML."]}
    out = await env.generate(batch)
    ao = _merge_actor_outputs(out, "smol")
    assert ao is not None
    assert len(ao.input_ids) == 1
    assert len(ao.input_ids[0]) == len(ao.attention_mask[0])
    decoded_text = tok.decode(ao.input_ids[0])
    assert "You are SmolLM" in decoded_text
    assert decoded_text.count("smol says:") == 0
    assert sum(ao.attention_mask[0]) == len(ao.input_ids[0])


@pytest.mark.slow
@pytest.mark.asyncio
async def test_one_actor_with_masking(smollm_actor):
    tok = smollm_actor.tokenizer
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=smollm_actor,
                system_prompt="You are SmolLM. Answer this question: ",
                sampling_params=SamplingParams(temperature=0.8),
            )
        ],
        round_spec="smol*2",
        reward_functions=[],
        run_concurrently=False,
        mask_other_agents_for_loss=True,
    )
    batch = {"text": ["Why is the sky blue?"]}
    out = await env.generate(batch)
    ao = _merge_actor_outputs(out, "smol")
    assert ao is not None
    assert len(ao.input_ids) == 1
    decoded_text = tok.decode(ao.input_ids[0])
    assert "You are SmolLM" in decoded_text
    assert decoded_text.count("smol says:") == 0
    len_tokens_to_mask = len(
        tok.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are SmolLM. Answer this question: Why is the sky blue?",
                }
            ],
            tokenize=True,
        )
    )
    assert sum(ao.attention_mask[0]) == len(ao.input_ids[0]) - len_tokens_to_mask
    assert all(m == 1 for m in ao.attention_mask[0][len_tokens_to_mask:])


@pytest.mark.slow
@pytest.mark.asyncio
async def test_two_actors_with_masking(smollm_actor, qwen_actor):
    tok1 = smollm_actor.tokenizer
    tok2 = qwen_actor.tokenizer
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=smollm_actor,
                system_prompt="You are SmolLM, try to convince qwen to help you write a GRPO implementation.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
            CollaborativeActorConfig(
                actor=qwen_actor,
                system_prompt="You are Qwen, the helpful assistant.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
        ],
        round_spec="smol -> qwen",
        reward_functions=[],
        run_concurrently=False,
        mask_other_agents_for_loss=True,
        prefill_name=True,
    )
    batch = {"text": [""]}
    out = await env.generate(batch)
    smol_ao = _merge_actor_outputs(out, "smol")
    qwen_ao = _merge_actor_outputs(out, "qwen")
    assert smol_ao is not None and qwen_ao is not None
    assert len(smol_ao.input_ids) == 1
    assert len(qwen_ao.input_ids) == 1
    smol_decoded = tok1.decode(smol_ao.input_ids[0])
    qwen_decoded = tok2.decode(qwen_ao.input_ids[0])
    assert "You are SmolLM" in smol_decoded
    assert "You are Qwen" in qwen_decoded
    assert qwen_decoded.count("smol says:") >= 1
    assert smol_decoded.count("qwen says:") >= 1
    smol_len_tokens_to_mask = len(
        tok1.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are SmolLM, try to convince qwen to help you write a GRPO implementation.",
                }
            ],
            tokenize=True,
        )
    )
    qwen_len_tokens_to_mask = len(
        tok2.apply_chat_template(
            [{"role": "system", "content": "You are Qwen, the helpful assistant."}],
            tokenize=True,
        )
    )
    assert all(m == 0 for m in smol_ao.attention_mask[0][:smol_len_tokens_to_mask])
    assert all(m == 0 for m in qwen_ao.attention_mask[0][:qwen_len_tokens_to_mask])
    count_tokens_smol = sum(smol_ao.attention_mask[0])
    count_tokens_qwen = sum(qwen_ao.attention_mask[0])
    assert all(
        m == 1
        for m in smol_ao.attention_mask[0][
            smol_len_tokens_to_mask : smol_len_tokens_to_mask + count_tokens_smol
        ]
    )
    assert all(m == 1 for m in qwen_ao.attention_mask[0][-count_tokens_qwen:])


@pytest.mark.slow
@pytest.mark.asyncio
async def test_two_actors_long_conversation(trump_actor, musk_actor):
    tok1 = trump_actor.tokenizer
    tok2 = musk_actor.tokenizer
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=musk_actor,
                system_prompt="You are Elon Musk. Argue with Trump about the Epstein list.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
            CollaborativeActorConfig(
                actor=trump_actor,
                system_prompt="You are Donald Trump.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
        ],
        round_spec="Musk -> Trump -> Musk -> Trump",
        reward_functions=[],
        run_concurrently=False,
        mask_other_agents_for_loss=True,
        prefill_name=True,
    )
    batch = {"text": [""]}
    out = await env.generate(batch)
    musk_ao = _merge_actor_outputs(out, "Musk")
    trump_ao = _merge_actor_outputs(out, "Trump")
    assert musk_ao is not None and trump_ao is not None
    assert len(musk_ao.input_ids) == 1
    assert len(trump_ao.input_ids) == 1
    trump_decoded = tok1.decode(trump_ao.input_ids[0])
    musk_decoded = tok2.decode(musk_ao.input_ids[0])
    assert "You are Elon" in musk_decoded
    assert "You are Donald" in trump_decoded
    assert trump_decoded.count("Musk says:") >= 2
    assert musk_decoded.count("Trump says:") >= 2


@pytest.mark.slow
@pytest.mark.asyncio
async def test_two_actors_concurrent(trump_actor, musk_actor):
    tok1 = trump_actor.tokenizer
    tok2 = musk_actor.tokenizer
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=musk_actor,
                system_prompt="You are Elon Musk. Argue with Trump about the Epstein list.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
            CollaborativeActorConfig(
                actor=trump_actor,
                system_prompt="You are Donald Trump.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
        ],
        round_spec="Musk -> (Trump/Musk)*3 -> Trump",
        reward_functions=[],
        run_concurrently=True,
        mask_other_agents_for_loss=True,
        prefill_name=True,
    )
    batch = {"text": [""] * 8}
    out = await env.generate(batch)
    musk_ao = _merge_actor_outputs(out, "Musk")
    trump_ao = _merge_actor_outputs(out, "Trump")
    assert musk_ao is not None and trump_ao is not None
    assert len(musk_ao.input_ids) == 8
    assert len(trump_ao.input_ids) == 8
    for i in range(8):
        assert len(musk_ao.input_ids[i]) == len(musk_ao.attention_mask[i])
        assert len(trump_ao.input_ids[i]) == len(trump_ao.attention_mask[i])
        trump_decoded = tok1.decode(trump_ao.input_ids[i])
        musk_decoded = tok2.decode(musk_ao.input_ids[i])
        assert "You are Elon" in musk_decoded
        assert "You are Donald" in trump_decoded
        assert trump_decoded.count("Musk says:") >= 1
        assert musk_decoded.count("Trump says:") >= 1


@conversation_reward_function("always_1")
def always_1(conversation, actor_name):
    return 1.0


@conversation_reward_function("dod_reward")
def dod_reward(conversation, actor_name):
    if actor_name == "Musk":
        return -1.0
    if actor_name == "Trump":
        return 2.0
    return 0.0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_reward_functions(trump_actor, musk_actor):
    tok1 = trump_actor.tokenizer
    tok2 = musk_actor.tokenizer
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=musk_actor,
                system_prompt="You are Elon Musk. Argue with Trump about the Epstein list.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
            CollaborativeActorConfig(
                actor=trump_actor,
                system_prompt="You are Donald Trump.",
                sampling_params=SamplingParams(temperature=0.8),
            ),
        ],
        round_spec="Musk -> Trump",
        reward_functions=[always_1, dod_reward],
        run_concurrently=False,
        mask_other_agents_for_loss=True,
    )
    batch = {"text": [""] * 4}
    out = await env.generate(batch)
    musk_ao = _merge_actor_outputs(out, "Musk")
    trump_ao = _merge_actor_outputs(out, "Trump")
    assert musk_ao is not None and trump_ao is not None
    assert len(musk_ao.input_ids) == 4
    assert len(trump_ao.input_ids) == 4
    trump_decoded = tok1.decode(trump_ao.input_ids[0])
    musk_decoded = tok2.decode(musk_ao.input_ids[0])
    assert "You are Elon" in musk_decoded
    assert "You are Donald" in trump_decoded
    assert len(musk_ao.rewards) == 4
    assert len(trump_ao.rewards) == 4
    assert len(trump_ao.reward_components.keys()) == 2
    assert len(musk_ao.reward_components.keys()) == 2
    assert trump_ao.reward_components["always_1"][0] == 1.0
    assert musk_ao.reward_components["always_1"][0] == 1.0
    assert trump_ao.reward_components["dod_reward"][0] == 2.0
    assert musk_ao.reward_components["dod_reward"][0] == -1.0
