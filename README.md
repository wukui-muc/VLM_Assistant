# VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving Visual-Language Models 
## IROS 2025

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
## Citation

If you use this paper in your research, please consider citing:

```bibtex
@misc{wu2025vlmgoodassistantenhancing,
      title={VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving Vision-Language Models}, 
      author={Kui Wu and Shuhang Xu and Hao Chen and Churan Wang and Zhoujun Li and Yizhou Wang and Fangwei Zhong},
      year={2025},
      eprint={2505.20718},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.20718}, 
}
