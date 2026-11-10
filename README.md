# video2deploymentG1

From a phone video to a Unitree G1 performing the motion.

Pipeline: TRAM video capture, SMPL to G1 retargeting (ASAP, PHC style), ASAP motion-tracking training in Isaac Gym with domain randomization, MuJoCo sim2sim check, rl_sar deployment on the G1.

Status: being set up. First motion documented: the dab (already deployed). Second motion in progress: the "6-7" gesture.

Companion repositories, pinned by commit in `forks.lock` (to come):
- ASAP fork: https://github.com/EricXieQ/ASAP, branch `dab-recipe-v6`, tag `dab-v6-recipe`
- rl_sar fork: https://github.com/EricXieQ/rl_sar_g1_deployment, branch `jetson-deploy`
- TRAM: https://github.com/yufu-wang/tram
