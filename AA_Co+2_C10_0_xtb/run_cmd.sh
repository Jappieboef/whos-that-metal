#!/bin/bash
conda activate seq2mSMILES_v2
xtb AA_Co+2_C10_0_xtb.xyz --chrg 2 --uhf 3 --ohess --gfn 2 > AA_Co+2_C10_0_xtb.out 2>&1
