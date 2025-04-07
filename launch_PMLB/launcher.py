import itertools
import subprocess
import argparse
import os
import json
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description="Launcher configuration.")
parser.add_argument("--n_jobs", type=int, required=True, help="How many runs?")
args = parser.parse_args()

config={
    'dataset':['197_cpu_act',
                '215_2dplanes',
                '227_cpu_small',
                '556_analcatdata_apnea2',
                '557_analcatdata_apnea1' ,
                '564_fried',
                '573_cpu_act',
                '218_house_8L',
                '225_puma8NH',
                '294_satellite_image',
                '666_rmftsa_ladata',
                '579_fri_c0_250_5',
                '586_fri_c3_1000_25',
                '590_fri_c0_1000_50',
                '593_fri_c1_1000_10',
                '595_fri_c0_1000_10',
                '654_fri_c0_500_10',
                '581_fri_c3_500_25',
                '582_fri_c1_500_25',
                '584_fri_c4_500_25'],
    'complexity':['cstd','cmod'],
    'pheno_freq':["[0,0,0]","[2,0,0]","[0,2,0]","[0,0,2]","[2,2,2]"],
    'genetic_mutations':[False,True],
    'feature_probability':[False,True],
    'seed':[2804,8356,16130,24305,24399],
    'n_jobs':[8]}

keys = list(config.keys())
combinations = list(itertools.product(*(config[key] for key in keys)))

# Exclude certain combos as before:
combinations = [
    combo for combo in combinations
    if not (dict(zip(keys, combo))['pheno_freq'] == "[0,0,0]" and
            dict(zip(keys, combo))['genetic_mutations'] is False)
]

# ------------------------------------------------------------------------------
# Optional step: If you'd like to skip combos that already appear in launched.txt,
# read them into a set first.
# ------------------------------------------------------------------------------
already_launched = set()
if os.path.exists("launched.txt"):
    with open("launched.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                already_launched.add(line)

already_terminated = set()
if os.path.exists("terminated.txt"):
    with open("terminated.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                already_terminated.add(line)

# Filter out combos that are already launched or terminated
remaining_combos = []
for combo in combinations:
    combo_dict = dict(zip(keys, combo))
    combo_str = json.dumps(combo_dict, sort_keys=True)
    if (combo_str not in already_launched) and (combo_str not in already_terminated):
        remaining_combos.append(combo)

def run_script(combo):
    """
    Runs the 'main.py' script with the given combo, logs the combo to
    'launched.txt' when launched, and logs to 'terminated.txt' if it finishes
    successfully.
    """
    args_dict = dict(zip(keys, combo))
    combo_str = json.dumps(args_dict, sort_keys=True)  # textual representation

    # --------------------------------------------------------------------------
    # 1) Log to launched.txt
    # --------------------------------------------------------------------------
    with open("launched.txt", "a") as f_launched:
        f_launched.write(combo_str + "\n")

    # Prepare the command
    cmd = [
        "python", "main.py",
        "--dataset", args_dict['dataset'],
        "--complexity", args_dict['complexity'],
        "--pheno_freq", args_dict['pheno_freq'],
        "--genetic_mutations", str(args_dict['genetic_mutations']),
        "--feature_probability", str(args_dict['feature_probability']),
        "--seed", str(args_dict['seed']),
        "--n_jobs", str(args_dict['n_jobs'])
    ]
    print(f"Launching: {' '.join(cmd)}")

    # Run and catch errors if desired
    try:
        subprocess.run(cmd, check=True)
        # ----------------------------------------------------------------------
        # 2) Log to terminated.txt only on successful completion
        # ----------------------------------------------------------------------
        with open("terminated.txt", "a") as f_terminated:
            f_terminated.write(combo_str + "\n")
    except subprocess.CalledProcessError as e:
        # If main.py exits with a non-zero status, you could handle that here
        print(f"Command failed: {cmd}")
        print(str(e))

# Parallel launch
max_parallel_jobs = args.n_jobs
Parallel(n_jobs=max_parallel_jobs)(
    delayed(run_script)(combo) for combo in remaining_combos
)
