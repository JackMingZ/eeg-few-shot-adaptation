import unittest

from src.real_eeg import EEGNet, parameter_counts


class ModelProfileTests(unittest.TestCase):
    def test_real_model_parameter_profile_is_explicit(self):
        counts = parameter_counts(EEGNet())
        self.assertEqual(counts, {'total': 2464, 'trainable': 2420, 'frozen': 44})


if __name__ == '__main__':
    unittest.main()
