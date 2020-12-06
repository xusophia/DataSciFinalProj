### Policy Gradient With Mean Rewards Baseline, Rewards-To-Go and Entropy Bonus

Base implementation of the policy gradient algorithm. This implementation uses `rewards-to-go` as the weight for the gradient.
Also, the model uses policy `entropy bonus` and state-specific `baseline`.

#### Run

1. Install dependencies: numpy, pandas, gym, pytorch, pybox2d
2. Run `python policy_gradient.py` for standard environment, `python policy_gradient.py --env input-noise` for Gaussian sensor input noise environment