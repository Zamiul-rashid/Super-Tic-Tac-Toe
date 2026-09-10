import csv
import json
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch
from pathlib import Path
from sttt.reports import new_report, summarize, write_report
from sttt.visualize import discover


class ReportTests(unittest.TestCase):
    def test_exports_and_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory/'checkpoint.pt'
            checkpoint.write_bytes(b'checkpoint identity')
            report = new_report(checkpoint, 4, 'tactical', 64, 7, 3)
            for i,(outcome,score) in enumerate([('wins',1),('draws',.5),('losses',0)]):
                report['games'].append({'game':i+1,'agent_side':'X' if i%2 == 0 else 'O',
                                        'outcome':outcome,'score':score,'moves':50+i,
                                        'seconds':1.,'agent_moves':25,'agent_search_seconds':.8})
            report['status'] = 'complete'
            path = write_report(report,directory)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['summary']['score_rate'],.5)
            self.assertEqual(saved['summary']['mean_moves'],51)
            self.assertEqual(saved['by_side']['X']['games'],2)
            self.assertEqual(saved['by_side']['O']['games'],1)
            with path.with_name(path.stem+'_games.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[0]['checkpoint_sha256'],saved['checkpoint_sha256'])
            self.assertEqual([r['outcome'] for r in rows],['wins','draws','losses'])
            with path.with_name(path.stem+'_summary.csv').open(newline='') as file:
                summary = list(csv.DictReader(file))[0]
            self.assertEqual(float(summary['score_rate']),.5)
            found,logs = discover([directory,path])
            self.assertEqual(len(found),1)
            self.assertEqual(logs,[])
            second = new_report(checkpoint,4,'random',64,7,3)
            self.assertNotEqual(second['run_id'],report['run_id'])

    def test_empty_and_partial(self):
        self.assertIsNone(summarize([])['score_rate'])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'checkpoint.pt'
            checkpoint.write_bytes(b'test')
            report = new_report(checkpoint,None,'random',8,0,10)
            report['status'] = 'interrupted'
            path = write_report(report,directory)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['summary']['games'],0)
            self.assertEqual(saved['status'],'interrupted')
            self.assertEqual(discover([directory]),([],[]))

    def test_evaluator_interrupt_saves_exports(self):
        from sttt.ai import evaluate
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'checkpoint.pt'
            checkpoint.write_bytes(b'test checkpoint')
            args = Namespace(checkpoint=str(checkpoint),seed=0,opponent='random',
                             simulations=8,games=10,output=directory)
            with patch('sttt.ai.load_model',return_value=(None,{'iteration':1})), \
                 patch('sttt.ai.search',side_effect=KeyboardInterrupt), patch('builtins.print'):
                evaluate(args)
            paths = list(Path(directory).glob('eval-*.json'))
            self.assertEqual(len(paths),1)
            data = json.loads(paths[0].read_text())
            self.assertEqual(data['status'],'interrupted')
            self.assertEqual(data['summary']['games'],0)
            self.assertEqual(len(list(Path(directory).glob('*.csv'))),2)

if __name__ == '__main__':
    unittest.main()
