import numpy as np
import subprocess

def main():
    # # Round 1
    # hyperparameters = [
    #     ('learning-rate', [3e-4, 1e-4, 3e-5], 3e-4),
    #     ('gamma', [0.98, 0.99, 0.995], 0.99),
    #     ('gae-lambda', [0.9, 0.92, 0.95, 0.97, 0.98], 0.95),
    #     ('clip-coef', [0.1, 0.15, 0.2, 0.25, 0.3], 0.2),
    #     ('ent-coef', [0.005, 0.01, 0.02, 0.03], 0.01),
    #     ('vf-coef', [0.3, 0.4, 0.5, 0.6, 0.7], 0.5),
    #     ('target-kl', [0.005, 0.01, 0.02, 0.03], 0.01),
    #     ('num_steps', [64, 128, 256], 128),
    #     ('num_envs', [4, 8, 16, 32], 8),
    #     ('update_epochs', [3, 4, 5, 6], 4),
    #     ('clip-vloss', [True, False], True),
    # ]
    # Round 1 best
    # learningrate0.0003_gamma0.98_gaelambda0.97_clipcoef0.25_entcoef0.01_vfcoef0.6_targetkl0.02_num_steps128_num_envs8_update_epochs3_clipvlossFalse
    # # Round 2
    hyperparameters = [
        ('learning-rate', [2e-4, 3e-4, 4e-4], 3e-4),
        ('gamma', [0.975, 0.980, 0.985], 0.98),
        ('gae-lambda', [0.965, 0.970, 0.975], 0.97),
        ('clip-coef', [0.225, 0.250, 0.275], 0.25),
        ('ent-coef', [0.0075, 0.0100, 0.0125], 0.01),
        ('vf-coef', [0.55, 0.60, 0.65], 0.6),
        ('target-kl', [0.015, 0.020, 0.025], 0.020),
        ('num_steps', [64, 128, 256], 128),
        ('num_envs', [4, 8, 16, 32], 8),
        ('update_epochs', [2, 3, 4], 3),
        ('clip-vloss', [True, False], True),
    ]
    # # Round 2 best
    # # learningrate0.0003_gamma0.98_gaelambda0.97_clipcoef0.25_entcoef0.01_vfcoef0.6_targetkl0.02_num_steps128_num_envs8_update_epochs3_clipvlossFalse
    total_timesteps = 300_000
    number_of_experiments = 10
    for n_iter in range(number_of_experiments):
        # Print a divider
        print()
        print()
        print()
        print("="*80)
        print(f"Experiment {n_iter+1}/{number_of_experiments}")
        print("="*80)
        # Create hyperparameter dictionary
        to_use = {}
        for name, choices, default in hyperparameters:
            # Find the position of name in hyperparameters
            # rank = 1 + [h[0] for h in hyperparameters].index(name)
            # prob = 1/(rank)**0.5
            # if np.random.rand() < prob:
            to_use[name] = np.random.choice(choices)
            # else:
            #     hyperparams[name] = default
            # If target-kl is None, then drop it
            if name == "target-kl" and to_use[name] is None:
                del to_use[name]
        # python multigrid/scripts/train_ppo_cleanrl.py --env-id MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obsticle --total-timesteps 100_000
        # Build the argument from the hyperparameter dictionary
        additional_cmds = []
        exp_name = ""
        for name, value in to_use.items():
            hyphen_name = name.replace("_", "-")
            short_name = name.replace("-", "")
            if short_name == "learningrate":
                short_name = "lr"
            additional_cmds.append(f"--{hyphen_name}")
            additional_cmds.append(str(value))
            exp_name += f"{short_name}{value}_"
        exp_name = exp_name[:-1] # Remove last underscore
        additional_cmds.append("--exp-name")
        additional_cmds.append(exp_name)
        # Build the command
        cmd = [
            "python",
            "multigrid/scripts/train_ppo_cleanrl.py",
            "--env-id",
            "MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obsticle",
            "--total-timesteps",
            f"{total_timesteps}",
        ]
        cmd.extend(additional_cmds)
        # Run the command
        result = subprocess.run(cmd)

        # Indicate success of failure
        if result.returncode == 0:
            print("Command executed successfully!")
        else:
            print(f"Command failed with return code {result.returncode}.")

if __name__ == "__main__":
    main()