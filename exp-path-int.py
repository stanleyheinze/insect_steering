import sys
sys.path.append('.')

import path_integrator_trial

for seed in range(0,50):
    for brain in ['nengo', 'sigmoid', 'ideal']:
        for low in ([0, 0.5] if brain!='ideal' else [0]):
            path_integrator_trial.PathIntegratorTrial().run(seed=seed, brain=brain,
                                                    verbose=False,
                                                    data_dir='paper-path-int',
                                                    data_format='npz',
                                                    path_output_rescale_low=low,
                                                    )
