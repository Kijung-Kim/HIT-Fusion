import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('reproduce',Path(__file__).parents[1]/'tools/reproduce.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class EvaluationTest(unittest.TestCase):
    def test_wrong_frame_is_both_fp_and_fn_and_mixed_is_unscored(self):
        gt={'a':{'kind':'impact','frames':[3,4]},'b':{'kind':'impact','frames':[8]},
            'c':{'kind':'impact','frames':[10]},'d':{'kind':'mixed_or_indeterminate'}}
        pred={k:{'status':'impact' if v>=0 else 'practice_gate','prediction_frame_id':v}
              for k,v in {'a':4,'b':9,'c':-1,'d':-1}.items()}
        r=module.evaluate(gt,pred)
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,2))
        self.assertEqual(r['localization_success'],1/3)
        self.assertEqual(r['verified_tn'],0)
        self.assertEqual(r['mixed_no_output'],1)
    def test_missing_sequences_cannot_silently_change_denominator(self):
        with self.assertRaises(ValueError):
            module.evaluate({'a':{'kind':'impact','frames':[1]}},{})

if __name__=='__main__':
    unittest.main()
