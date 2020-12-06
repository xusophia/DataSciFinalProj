# LunarLander - Reinforcement Learning


Final project for EE460J Data Science Lab

@TODO: DEFINE DQN and VPG acronyms just for clarity somewhere

## Introduction 
- Outline goal of project
- Open AI gym - how we got our 'data'
- define 'lander' if that's what we're gonna use to describe lander a lot

## Models
- Vanilla DQN
- Double DQN
- Dueling DQN 
- Vanilla Policy Gradient 

## Noise in the Environment

We wanted to test the robustness of our various DQN/VPG solutions to Lunar Lander in real-world scenarios.

3 types of noise we tried to add, according to [this paper](https://arxiv.org/pdf/2011.11850.pdf):
1. Action Noise - In the context of Lunar Landing, this is engine failure
2. Turbulence
3. State Noise
  The state vector consists of 8 state variables, as seen in the Open AI Lunar Lander code, as it consists of:
  ```             s[0] is the horizontal coordinate
                  s[1] is the vertical coordinate
                  s[2] is the horizontal speed
                  s[3] is the vertical speed
                  s[4] is the angle
                  s[5] is the angular speed
                  s[6] 1 if first leg has contact, else 0
                  s[7] 1 if second leg has contact, else 0         
  ```
  In our implementation, we added random noise to the horizontal coordinate `s[0]`, vertical coordinate `s[1]`, and the angle `s[4]` of the lunar lander.
   
   - outline partially Observable Markov Decision Process 
   
   ## Conclusions
