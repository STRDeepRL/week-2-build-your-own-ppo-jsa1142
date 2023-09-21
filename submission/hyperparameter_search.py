import numpy as np
import subprocess

def main():
    # Round 1
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
    # learningrate0.0001_gamma0.995_gaelambda0.9_clipcoef0.25_entcoef0.005_vfcoef0.3_targetkl0.03_num_steps64_num_envs8_update_epochs5_clipvlossTrue
    # Round 2
    hyperparameters = [
        ('learning-rate', [2e-4, 1e-4, 0.5e-4], 1e-4),
        ('gamma', [0.9925, 0.995, 0.9975], 0.995),
        ('gae-lambda', [0.89, 0.9, 0.91], 0.90),
        ('clip-coef', [0.225, 0.25, 0.275], 0.25),
        ('ent-coef', [0.0025, 0.0050, 0.0075], 0.005),
        ('vf-coef', [0.25, 0.3, 0.35], 0.3),
        ('target-kl', [0.025, 0.030, 0.035, None], 0.03),
        ('num_steps', [32, 64, 128], 64),
        ('num_envs', [4, 8, 16, 32], 8),
        ('update_epochs', [3, 4, 5, 6], 5),
        ('clip-vloss', [True, False], True),
    ]
    total_timesteps = 300_000
    number_of_experiments = 10
    for n_iter in range(number_of_experiments):
        print()
        print()
        print()
        print("="*80)
        print(f"Experiment {n_iter+1}/{number_of_experiments}")
        print("="*80)
        # Sample hyperparameters
        hyperparams = {}
        for name, choices, default in hyperparameters:
            # Find the position of name in hyperparameters
            # rank = 1 + [h[0] for h in hyperparameters].index(name)
            # prob = 1/(rank)**0.5
            # if np.random.rand() < prob:
            hyperparams[name] = np.random.choice(choices)
            # else:
            #     hyperparams[name] = default
            # If target-kl is None, then drop it
            if name == "target-kl" and hyperparams[name] is None:
                del hyperparams[name]
        # python multigrid/scripts/train_ppo_cleanrl.py --env-id MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obsticle --total-timesteps 100_000
        additional_cmds = []
        exp_name = ""
        for name, value in hyperparams.items():
            hyphen_name = name.replace("_", "-")
            short_name = name.replace("-", "")
            additional_cmds.append(f"--{hyphen_name}")
            additional_cmds.append(str(value))
            exp_name += f"{short_name}{value}_"
        exp_name = exp_name[:-1] # Remove last underscore
        additional_cmds.append("--exp-name")
        additional_cmds.append(exp_name)
        cmd = [
            "python",
            "multigrid/scripts/train_ppo_cleanrl.py",
            "--env-id",
            "MultiGrid-CompetativeRedBlueDoor-v2-DTDE-Red-Single-with-Obsticle",
            "--total-timesteps",
            f"{total_timesteps}",
        ]
        cmd.extend(additional_cmds)
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("Command executed successfully!")
        else:
            print(f"Command failed with return code {result.returncode}.")

if __name__ == "__main__":
    main()