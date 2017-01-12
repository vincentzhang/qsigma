# This repo holds the code for Qsigma project.

## For now, it gathers all the original code for the reinforcement learning project in the course CMPUT609.
**Team members: Vincent (Zichen) Zhang, Kristopher De Asis.**

├── README.md     : this file  
├── MDPy.py       : base module for MDP  
├── agent.py      : base module for agent  
├── Qsigma.py     : module for Q(sigma) agent  
├── SARSA.py      : module for SARSA agent  
├── expSARSA.py   : module for Expected SARSA agent  
├── treebackup.py : module for Tree Backup agent  
├── randomwalk.py : module for the random walk MDP  
├── example  
│   ├── Qsig_test.py      : script to produce all data need for plot_rms_err.m  
│   ├── plot_randomwalk.py: produce plots for figures 5,6 in the report  
│   └── plot_rms_err.m    : matlab script to produce figures 1-4 in the report

Data Files needed for running plot_randomwalk.py  
├── example  
│   ├── rms_nstep_Q_10_Qsigma.p  
│   ├── rms_nstep_Q_10_Sarsa.p  
│   ├── rms_nstep_Q_10_TreeBackup.p  
│   ├── rms_nstep_Q_10_expSarsa.p  
│   ├── rms_nstep_Q_50_Qsigma.p  
│   ├── rms_nstep_Q_50_Sarsa.p  
│   ├── rms_nstep_Q_50_TreeBackup.p  
│   └── rms_nstep_Q_50_expSarsa.p  
