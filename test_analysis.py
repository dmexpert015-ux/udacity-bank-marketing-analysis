"""Checks source integrity and the properties that make the model evaluation valid."""
import unittest
from analysis import load_data, rate_table, fit_model

class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.metrics = fit_model(cls.df)

    def test_source_and_target(self):
        self.assertEqual(len(self.df), 45211)
        self.assertEqual(int(self.df.subscribed.sum()), 5289)
        self.assertEqual(self.df.duplicated().sum(), 0)

    def test_groups_reconcile(self):
        table = rate_table(self.df, 'poutcome')
        self.assertEqual(int(table['count'].sum()), len(self.df))
        self.assertEqual(int(table['sum'].sum()), 5289)
        self.assertEqual(int(table.loc['success', 'sum']), 978)

    def test_temporal_split_and_no_duration(self):
        self.assertEqual(self.metrics['train_rows'], 36168)
        self.assertEqual(self.metrics['test_rows'], 9043)
        self.assertIn('duration', self.metrics['excluded_columns'])
        self.assertIn('campaign', self.metrics['excluded_columns'])

    def test_ranking_beats_uninformative_baseline(self):
        self.assertGreater(self.metrics['roc_auc'], .5)
        self.assertGreater(self.metrics['average_precision'], self.metrics['test_subscription_rate'])
        self.assertGreater(self.metrics['top_20_percent_lift'], 1)
        self.assertEqual(self.metrics['baseline_recall'], 0)

if __name__ == '__main__':
    unittest.main()
