import unittest
import tempfile
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch

from src.real_eeg import EEGNet, make_split, configure_adaptation, seed_all, load_data


class ProtocolTests(unittest.TestCase):
    def test_incorrect_units_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            np.savez(Path(root)/'subject_01.npz',X=np.zeros((4,22,500)),y=np.arange(4),
                     session=np.array(['0train']*4),trial_id=np.array(['a','b','c','d']),
                     metadata=json.dumps({'unit':'volt','sampling_rate':125,'filter_hz':[4,38]}))
            with self.assertRaisesRegex(ValueError,'metadata'):
                load_data(root,[1])

    def test_duplicate_subjects_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError,'duplicates'):
            load_data('not_needed',[1,2,2])

    def test_nested_budget_and_subject_session_isolation(self):
        subjects=np.repeat([1,2,3],96)
        sessions=np.tile(np.repeat(['0train','1test'],48),3)
        y=np.tile(np.repeat(np.arange(4),12),6)
        train,val,pool,test,ordered,held=make_split(y,subjects,sessions,1,7,10)
        self.assertEqual(held,2)
        self.assertEqual(set(subjects[train]),{3})
        self.assertEqual(set(subjects[val]),{2})
        self.assertEqual(set(sessions[pool]),{'0train'})
        self.assertEqual(set(sessions[test]),{'1test'})
        self.assertEqual(len(set(np.concatenate([train,val,pool,test]))),len(y))
        small=set(np.concatenate([a[:1] for a in ordered]))
        large=set(np.concatenate([a[:10] for a in ordered]))
        self.assertTrue(small <= large)
        self.assertTrue(large.isdisjoint(test))

    def test_adaptation_does_not_mutate_frozen_state(self):
        torch.set_num_threads(2)
        for method in ['head','full','affine_head','scale_head']:
            seed_all(7)
            model=EEGNet()
            before=deepcopy(model.state_dict())
            configure_adaptation(model,method)
            params=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(params,lr=.001)
            logits=model(torch.randn(4,1,22,500))
            self.assertEqual(tuple(logits.shape),(4,4))
            torch.nn.functional.cross_entropy(logits,torch.arange(4)).backward()
            optimizer.step(); model.constrain()
            changed=[]
            for name,value in model.state_dict().items():
                parameter=dict(model.named_parameters()).get(name)
                if parameter is None or not parameter.requires_grad:
                    self.assertTrue(torch.equal(value,before[name]),(method,name))
                elif not torch.equal(value,before[name]):
                    changed.append(name)
            self.assertIn('head.weight',changed)
            if method in ['affine_head','scale_head']: self.assertIn('scale',changed)
            if method=='affine_head': self.assertIn('offset',changed)

    def test_insufficient_calibration_is_rejected(self):
        y=np.tile(np.arange(4),6)
        subjects=np.repeat([1,2,3],8)
        sessions=np.tile(np.repeat(['0train','1test'],4),3)
        with self.assertRaises(ValueError): make_split(y,subjects,sessions,1,7,10)


if __name__=='__main__': unittest.main()
