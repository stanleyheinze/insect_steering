import sys
sys.path.append('.')

import path_integrator_trial

path_integrator_trial.PathIntegratorTrial().run(
    seed=1, 
    brain='nengo',
    plt=True,
    data_dir=None,
)
