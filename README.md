## Introduction

This is a pytorch implementation of [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://arxiv.org/abs/1706.02275) algorithm.
Original implementer by [xuehy](https://github.com/xuehy). Modified for compatibility with Battle Royale environment.

## Environment

[Battle Royale](https://github.com/adhipradhana/BattleRoyale) environment is used in this project. 
- Unity Environment wrapped in OpenAI Gym. 
- Fully competitive environment.
- Supports multi agents.

## Dependency

- [pytorch](https://github.com/pytorch/pytorch)
- [visdom](https://github.com/facebookresearch/visdom)
- Python 3.7.6 (recommend using the anaconda/miniconda)