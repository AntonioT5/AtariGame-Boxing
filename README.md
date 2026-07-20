# Atari Game Boxing

This project is focused training a neural networks agent to become amazing Atari Boxing player. We utilize a Double DQN combine with a pre-trained ResNet architecture. For simplicity and structure development, this project is implemented across several paheses.

## Phase 0: Setup and Environment
Phase 0 is strictly phase in which we download the required libraries and set up WSL: Ubuntu in VS Code. We use WSL: Ubuntu because of the fact that some RL libraries are very difficult to install on Windows, and using WSL:Ubuntu makes RL development much easer.

## Phase 1: Preprocessing pipline
In this phase, we resize the whole Atari Boxing game frames from (210, 160, 3) to (84, 84, 1) with help of *preprocessing.py* script. This script makes frames smaller and removes unnecessary RGB channels. Because the network has fewer pixels to process, training is significantly faster. Finally, we test the setup usign *test-env.py* to ensure the environment is working correctly before moving on.

## Phase 2: DQN from scratch
Initially, I attempted to solve the environment using a standard DQN, but the results were poor. The agent suffered from action collapse-meaning it would just pick one out of the 18 actions and stick to it endlessly. After some experimentation, I upgrade the architecture to Double DQN and the results immediately were better. 

Our custom architecture uses 3 *Conv2d* layers, followed by a fully connected *Linear* output layer that provides Q-values for the 18 actions. The core logic of this script builds upon the auditory and laboratory exercise from the classes. For this project is implemented in 
*boxing_double_dqn_agent.py*. 

To train and evaluate the Double DQN, we created two scripts:
- *train_boxing_dqn.py*:
    This script handles the training loop for the Double DQN and store the model weights in a directory called *LeanedExperience*. A new checkpoint saved every 50 episodes, allowing us to load them later and measure the boxer's performance over time. Because Atari Boxing is two-player game, we set Agent1 to use our Double DQN and Agent2 to take random actions. I initially tried training Agent1 against Agent2 using Double DQN for both, but the neural networks could not understand which actions were good. Therefore, thraing agent agenst a random agent were better choice. Hyperparameters for traing were: 
    - max steps per episodes 2000
    - target network update every 2000 steps
    - train every 16 steps
    - min memory size to start traing 10 000
    - Epsilon starts at 1.0 and decays by a factor of 0.999997 every step
    At the end of the episode, the console prints out the rewrds taken by agent1 and agent2, value of the epsilon and win-rate for every 50 episodes.
- *boxingDQN_vs_randomAction.py*:
    This script is built only for evaluation. It test the trained agent agains a random opponent. It calculates the distribution od the actions the agent choose (the percentage of the time each action is used), as well as the final win, loss and draw rates.

## Phase 3: Pretrained Network
For this phase, I made a decission to use ResNet pre-trained architecture.To adapt the standard ResNet to our specific environment, I modified the initial Conv2d layer to accept 4 input channels instead of its default 3 channels. Additionally, I replaced the final fully connected layer with a custom Linear layer that support output of 18 actions. 

Initially, training this network was challenging. The standard training loop was overwritting the pre-trained weights, causing the model to lose already learned skills by ResNet. To solve this issue, I froze the weights of the layer1, layer2 and layer3 of the ResNet network since I tested with freezing the layer1 and layer2 which got me worst results. The modified architecture is implemented in *boxing_pretrained_dqn_agent.py*.

The traing and testing architecture follow the exact same concepts established in Phase 2. The names od the scripts are *train_boxing_pretrained_dqn.py* and *pretrainedBoxingDQN_vs_randomAction.py*.




