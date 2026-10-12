import argparse
from pathlib import Path
import tempfile
import unittest

from audit_real_results import audit
from run_real_pipeline import build_commands
from src.experiment_profiles import PROFILES
from src.real_eeg import EEGNet, configure_adaptation, parameter_counts


class PublicWorkflowTests(unittest.TestCase):
    def test_audit_rejects_empty_input_without_success_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No completed runs'):
                audit(tmp)
            self.assertFalse((Path(tmp)/'audit.json').exists())

    def test_profiles_match_preprocessing_and_training(self):
        for name, profile in PROFILES.items():
            args = argparse.Namespace(profile=name, subjects=[1,2,3], targets=[1],
                                      seeds=[7], provider='nemar', epochs=5,
                                      adapt_epochs=5, budgets=[1,3], output=None, no_plots=True)
            _, prep, train = build_commands(args)
            for flag in ('--sampling-rate','--epoch-start','--epoch-end'):
                self.assertEqual(prep[prep.index(flag)+1], train[train.index(flag)+1])
            samples = int(train[train.index('--samples')+1])
            self.assertEqual(samples, round(profile.sampling_rate*(profile.epoch_end-profile.epoch_start)))
            self.assertEqual(Path(train[train.index('--data-root')+1]).name, prep[prep.index('--output-dir')+1])
            self.assertIn('--no-plots', train)

    def test_main_profile_adaptation_counts(self):
        model = EEGNet(samples=256, temporal_kernel=32, dropout=0.25)
        self.assertEqual(parameter_counts(model), {'total':1760,'trainable':1716,'frozen':44})
        for method, count in [('head',516),('full',1716),('scale_head',538),('affine_head',560)]:
            configure_adaptation(model,method)
            self.assertEqual(parameter_counts(model)['trainable'],count)
