# VLM_Assistant
VLM assistant to help visual tracking agent recovery from failure. 

## Dependencies
 - Unrealzoo
 - OpenAI
 - Tracking-anything-with-DEVA
## Installation

### Unrealzoo
```
git clone https://github.com/UnrealZoo/unrealzoo-gym.git
cd unrealzoo-gym
pip install -e .

#load environment binary from the official website, then set the environment variable before running.
export UnrealEnv={path-to-your-binary}
```
### Tracking-anything-with-DEVA
follow the [official guide](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)  
(The unrealzoo project could also simulate mask image in virtual environment by changing the observation type to ``Mask")

## Run example
before running, make sure each dependency could be imported correctly. 
```
python VLMFailureRecovery_reflection_deva_goal_conditioned_bbox.py --env UnrealTrack-Old_Factory_01-ContinuousColor-v0
```
