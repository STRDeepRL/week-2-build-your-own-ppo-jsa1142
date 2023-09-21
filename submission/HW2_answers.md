### Task 1 Questions
After running the above command, observe the outputs in the command line. This will provide essential information required to train your RL agent.


#### Questions for General Deep RL Training Parameters Understanding
**Q.1** From the command line outputs, can you report the values for the following parameters from the command line outputs? Additionally, please describe the role of each parameter in the training loop and explain how these values influence training in a sentence or two. This exercise can help you grasp the fundamentals of `Sample Efficiency` and understand the tradeoffs when scaling your training process in a parallel fashion.  

#### Answer:

- **num_envs**: 4
  - The number of different random seeds used to generate environments for training.
- **batch_size**: 512
  - Total number of time steps collected in a single rollout: `batch_size = num_steps * num_envs`.
- **num_minibatches**: 4
  - Number of minibatches. This divides the `batch_size` to give the minibatch size: `minibatch_size = batch_size / num_minibatches`.
- **minibatch_size**: 128
  - The number of time steps used to compute the policy loss. This is computed as `minibatch_size = batch_size / num_minibatches`.
- **total_timesteps**: 10,000,000
  - Total number of time steps to train the agent for. This is the number of environment steps sampled in total across all environments.
- **num_updates**: 19,531
  - Total number of training updates. This is computed as `num_updates = total_timesteps / batch_size = 10,000,000 // 512 = 19,531`. 
- **num_steps**: 128
  - The horizon (number of time steps) for each rollout.
- **update_epochs**: 4
  - Number of times to reuse each batch for training. 

## Task 2 - Understand the Deep RL Training Loop Dataflow & Implement Techniques to Minimize Learning Variance

In this task, you will delve into the specifics of the vectorized training architecture, which consists of two pivotal phases: the `Rollout Phase` and the `Learning Phase`. This is the parallelized training architecture that many Deep RL algorithms, including PPO used. You will also explore the techniques employed by PPO to reduce variance in learning, particularly focusing on the Generalized Advantage Estimation (GAE). You will enhance your understanding by identifying these phases in the code and implementing GAE to reduce variance of the training data before the `Learning Phase` when using the diversed data collected from the `Rollout Phase`.

### Questions to Enhance Understanding of the Deep RL Training Loop
***Q.1*** As mentioned in [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), PPO employs a streamlined paradigm known as the vectorized architecture. This architecture encompasses two phases within the training loop:

- **Rollout Phase**: During this phase, the agent samples actions for 'N' environments and continues to process them for a designated 'M' number of steps.

- **Learning Phase**: In this phase, fundamentally, the agent learns from the data collected during the rollout phase. This data, with a length of NM, includes 'next_obs' and 'done'.

Utilizing your baseline codebase tagged `v2.1`, please pinpoint the `Rollout Phase` and the `Learning Phase` within the codebase, indicating specific line numbers. 

* For instance, the lines [189-211 in CleanRL ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py#L189-L211) represent the Rollout Phase in their PPO implementation.  

### Answer:
- **Rollout Phase**: [lines 500-528](multigrid/scripts/train_ppo_cleanrl.py)
- **Learning Phase**: [lines 620-707](multigrid/scripts/train_ppo_cleanrl.py)

## Task 3 - Tuning the 🎲 **Exploration & Exploitation Strategies** using Algorithm-Specific Hyperparameters

### Deepening Your Understanding to Interpret Your Results
***Q.1*** Train a baseline agent using default or adjusted parameter values. Capture and present Tensorboard screenshots to report the following training metrics. Indicate the `Sample Efficency`, the number of training timesteps and policy updates, required to achieve the Training Baseline Thresholds:

Only running ~300k steps does not hit the Baseline Thresholds.

![Baseline metrics](baseline_metrics.png)


**CleanRL Agent Training Baseline Thresholds for Your Reference**:
- `episodic_length` should converge to a solution within 40 time steps and maintain for at least 100k time steps at the end of training.
- `episodic_return` should converge to consistently achieve 2.0+ returns, enduring for a minimum of the last 100k time steps.
- `explained_variance` should stabilize at a level above 0.6 for at least the last 100k time steps.
- `entropy` should settle at values below 0.3 for a minimum of 100k final training steps.

### Hands-on Experiences on PPO-Specific Hyperparameters
***Q.2*** If your baseline agent struggles to achieve the Training Baseline Thresholds, or if there's potential for enhancment, now you are getting the chance to fine-tuning the following PPO-specific parameters discussed in class to improve the performance of your agent. You may want to run multiple versions of experinements, so remember to modify `--exp-name` to differentiate between agent configurations. For final submissions, pick the top 3 performing or representable results and present the training metrics via screenshots and specify the number of timesteps and policy updates needed to fulfill or surpass the Training Baseline Thresholds. (Including links to their videos will be ideal)

- **gamma** (default = 0.99)
- **gae-lambda** (default = 0.95)
- **clip-coef** (default = 0.2)
- **clip-vloss** (default = True)
- **ent-coef** (default = 0.01)
- **vf-coef** (default = 0.5)
- **target-kl** (default = None)

Additionally, consider tweaking the following generic Deep RL hyperparameters:

- **num_envs** (default = 8)
- **batch_size** (default = int(args.num_envs * args.num_steps) = 1024)
- **num_minibatches** (default = 4)
- **minibatch_size** (default = int(args.batch_size // args.num_minibatches) = 256)
- **total_timesteps** (default = 10_000_000)
- **num_updates** (default = args.total_timesteps // args.batch_size = 9765)
- **num_steps** (default = 128)
- **update_epochs** (default = 4)

I run a three-stage hyperparameter search in [the hyperparameter search script](./hyperparameter_search.py).

The only run that came close to possibly hitting the benchmarks was 
`--learning-rate 0.0003 --gamma 0.98 --gae-lambda 0.97 --clip-coef 0.25 --ent-coef 0.01 --vf-coef 0.6 --target-kl 0.02 --num-steps 128 --num-envs 8 --update-epochs 3 --clip-vloss False`
and this only ran for 300k steps, so did not get close to the benchmarks.
Re-running this model for 10M steps hits the benchmark. This is hitting the benchmark around 2M steps

![CleanRL Training](cleanrl_training.png)

![Hyperparameter search](hp_search.png)

**Suggestion for future courses**: Performing a long hyperparameter search is not fun or a rewarding learning experience for the student.
It would have been more rewarding to give us optimal hyperparameters as a way to verify the changes to the code were correct.
That way we could go through and modify our code, and check if our changes were correct by whether or not the changes resulted in runs that were competitive with the ideal run.
That would also make further refinements to the hyperparameters more interesting as we would be starting at the baseline.

## Task 4: Bring the Lessons Learned from CleanRL to RLlib to solve a 1v1, 🤖 🆚 🤖 Scenario 

As you get familiar with PPO by working through the CleanRL implementation, let's pivot back to RLlib. We'll harness our understanding of hyperparameter tuning to address a 1v1 competition with a pre-trained opponent.

### Q.1 Metrics to Report:

As the same as Task 2&3, document the following training metrics, showcasing them with screenshots. Also, detail the number of timesteps and policy updates that meet or exceed the Training Baseline Thresholds.

**Suggestion for future courses** Give the regex in the same location as the list of metrics. I spent time converting the markdown list to regex before realizing it was in the "tips" section.

Here are the metrics from the run:

![RLlib Metrics 1](rllib_metrics_1.png)
![RLlib Metrics 2](rllib_metrics_2.png)
![RLlib Metrics 3](rllib_metrics_3.png)

The model is converging around 600k steps, as seen in the metrics discussed for the baseline (plus a few plots from in-exact regex and repeats):

![Converging model](1v1_converging_model.png)