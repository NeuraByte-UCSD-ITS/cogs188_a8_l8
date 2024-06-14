# cogs188_a8_l8
A8_L8 assignment Repository

Note: could not complete part 4 due to limited computational resources (the code has been running for several days locally and on google collab)


# Instructions to run code

### 1. in file "lunar_lander_dqn.py" modify path in line 14 to a path output on your system for dqn_qlearning_sarsa folder to be stored

- Line 14: project_base_path = "/Users/computername/desiredOutputPath" 

### 2. in file "cartpole_algorithms.py" modify path in line 16 to a path output on your system for dqn_a2c_ppo folder to be stored

- Line 16: project_base_path = "/Users/computername/desiredOutputPath" 

### 3. install all packages/dependencies 

- #### 3.a) open terminal, cd into part 2 folder `cd part2` and execute the following commands in the `part2` directory of the project
    - python -m venv a8_l8
    - source a8_l8/bin/activate     
    - pip install -r requirements.txt

### 4. run lunar lander training script and generate `dqn_qlearning_sarsa` folder & 3 metrics results 
    - python lunar_lander_dqn.py

### 5. run evaluation script to generate average reward metric results and all model evaluations 
    - python algorithms_evaluation.py

### 7. Execute code in report_part3.ipynb to run the code and generate the final results for report_part3

### 8. run cartpole training script and generate `cartpole_project` folder & 3 metrics results for A2C, PPO, DQN implementation 
    - python cartpole_algorithms.py

### Note: this requires large computation resources, I ran the script for several days in both local and google doc and it is still running, thus unable to complete implementation of new algorithms on the new environments and do report4. If you choose to run code on google collab go to "cartpole_algorithms.py" file and comment lines 16-19 and uncomment lines 22-26 to save output to your google drive or uncomment lines 30-33 to save outputs to local on google collab

### 9. run evaluation script to generate average reward metric results and all (A2C, PPO, DQN) model evaluations 
    - python cartpolealgothinms_evaluation.py


