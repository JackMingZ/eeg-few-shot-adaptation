"""Malformed matrices must fail even if their row counts look correct."""
from copy import deepcopy
import unittest

import pandas as pd

from export_public_results import (
    BUDGETS, METHOD_BUDGETS, PROFILE, RUNS, SEEDS, SUBJECTS, TRAINING,
    validate_matrix,
)


class ExportMatrixTests(unittest.TestCase):
    def setUp(self):
        self.configs = [dict(PROFILE, **TRAINING, target=t, seed=s,
                             status='complete', skip_csp=False, input_scale_microvolt=1.0,
                             validation_subject=t % 9 + 1,
                             adaptation_batchnorm='frozen source statistics for all methods')
                        for t, s in sorted(RUNS)]
        self.frame = pd.DataFrame(
            [(t,s,m,b) for t,s in sorted(RUNS) for m,b in sorted(METHOD_BUDGETS)],
            columns=['target','seed','method','budget_per_class'])
        self.plan = dict(PROFILE, **TRAINING, targets=SUBJECTS, seeds=SEEDS)

    def test_complete_matrix_accepted(self):
        validate_matrix(self.configs, self.frame, self.plan)

    def test_same_size_wrong_seed_or_method_budget_rejected(self):
        for key, value in [('seed', 99), ('target', 10), ('method','unknown'), ('budget_per_class', 2)]:
            with self.subTest(key=key):
                frame = self.frame.copy()
                frame.loc[0,key] = value
                with self.assertRaisesRegex(ValueError, 'matrix mismatch'):
                    validate_matrix(self.configs, frame, self.plan)

    def test_duplicate_and_missing_rows_rejected(self):
        for frame in (self.frame.iloc[1:], pd.concat([self.frame.iloc[1:],self.frame.iloc[[1]]])):
            with self.assertRaises(ValueError):
                validate_matrix(self.configs, frame, self.plan)

    def test_config_drift_rejected(self):
        for key,value in [('epochs',5),('adapt_lr',0.003),('adapt_epochs',25),
                          ('patience',2),('budgets',[1,3]),('input_scale',100),
                          ('skip_csp',True),('euclidean_alignment',True),('samples',500)]:
            with self.subTest(key=key):
                configs=deepcopy(self.configs)
                configs[0][key]=value
                with self.assertRaises(ValueError):
                    validate_matrix(configs,self.frame,self.plan)

    def test_duplicate_run_configuration_rejected(self):
        configs=deepcopy(self.configs)
        configs[0]=configs[1]
        with self.assertRaisesRegex(ValueError, 'target/seed'):
            validate_matrix(configs,self.frame,self.plan)

    def test_batch_plan_must_match_runs(self):
        plan=deepcopy(self.plan)
        plan['adapt_lr']=0.003
        with self.assertRaisesRegex(ValueError, 'Batch plan'):
            validate_matrix(self.configs,self.frame,plan)
