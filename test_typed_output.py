#!/usr/bin/env python3
"""Quick test to check if the new typed environment output works."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from actors.environments.types import EnvironmentOutput, ActorOutput

# Test creating the new typed outputs
def test_typed_output():
    print("Testing new typed environment output...")
    
    # Create mock data
    input_ids = [[1, 2, 3, 4], [5, 6, 7, 8]]
    attention_mask = [[1, 1, 1, 1], [1, 1, 1, 1]]
    total_rewards = [0.5, -0.2]
    reward_components = {
        "length_penalty": [-0.1, -0.3],
        "quality_score": [0.6, 0.1]
    }
    
    # Create ActorOutput
    actor_output = ActorOutput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        total_rewards=total_rewards,
        reward_components=reward_components
    )
    
    print(f"ActorOutput created successfully!")
    print(f"  Input IDs: {actor_output.input_ids}")
    print(f"  Total rewards: {actor_output.total_rewards}")
    print(f"  Reward components: {actor_output.reward_components}")
    
    # Create EnvironmentOutput
    env_output = EnvironmentOutput(
        actors={"main": actor_output}
    )
    
    print(f"EnvironmentOutput created successfully!")
    print(f"  Actors: {list(env_output.actors.keys())}")
    print(f"  Main actor total rewards: {env_output.actors['main'].total_rewards}")
    print(f"  Main actor reward components: {env_output.actors['main'].reward_components}")
    
    return True

if __name__ == "__main__":
    try:
        test_typed_output()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
