import unittest
import sys
sys.path.append("..")
import model
import torch

class TestModel(unittest.TestCase):

    def test_get_k_best_predicted_antecedents(self):
        antecedent_idx = [[1, 2, 3, 4], [5, 6, 7, 8]]
        antecedent_scores = [[0, 20, 50, 30, 40], [0, 60, 30, 80, 40]]
        span_starts = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
        span_ends = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
        k = 3
        expected_k_best_antecedent_scores = torch.tensor([[50, 40, 30], [80, 60, 40]])
        expected_k_best_antecedent_idx = torch.tensor([[2, 4, 3], [7, 5, 8]])
        expected_k_best_antecedent_starts = torch.tensor([[1002, 1004, 1003], [1007, 1005, 1008]])
        expected_k_best_antecedent_ends = torch.tensor([[2002, 2004, 2003], [2007, 2005, 2008]])
        
        k_best_antecedent_idx, k_best_antecedent_scores, k_best_antecedent_starts, k_best_antecedent_ends = model.CorefModel.get_k_best_predicted_antecedents(antecedent_idx, antecedent_scores, span_starts, span_ends, k)
        self.assertTrue(torch.equal(expected_k_best_antecedent_scores, k_best_antecedent_scores))
        self.assertTrue(torch.equal(expected_k_best_antecedent_idx, k_best_antecedent_idx))
        self.assertTrue(torch.equal(expected_k_best_antecedent_starts, k_best_antecedent_starts))
        self.assertTrue(torch.equal(expected_k_best_antecedent_ends, k_best_antecedent_ends))

    def test_get_k_best_predicted_antecedents_dummy(self):
        antecedent_idx = [[1, 2, 3, 4], [5, 6, 7, 8]]
        antecedent_scores = [[0, 20, 50, 30, 40], [0, 60, -10, 80, -20]]
        span_starts = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
        span_ends = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]
        k = 3
        expected_k_best_antecedent_scores = torch.tensor([[50, 40, 30], [80, 60, 0]])
        expected_k_best_antecedent_idx = torch.tensor([[2, 4, 3], [7, 5, -1]])
        expected_k_best_antecedent_starts = torch.tensor([[1002, 1004, 1003], [1007, 1005, -1]])
        expected_k_best_antecedent_ends = torch.tensor([[2002, 2004, 2003], [2007, 2005, -1]])

        k_best_antecedent_idx, k_best_antecedent_scores, k_best_antecedent_starts, k_best_antecedent_ends = model.CorefModel.get_k_best_predicted_antecedents(antecedent_idx, antecedent_scores, span_starts, span_ends, k)
        self.assertTrue(torch.equal(expected_k_best_antecedent_scores, k_best_antecedent_scores))
        self.assertTrue(torch.equal(expected_k_best_antecedent_idx, k_best_antecedent_idx))
        self.assertTrue(torch.equal(expected_k_best_antecedent_starts, k_best_antecedent_starts))
        self.assertTrue(torch.equal(expected_k_best_antecedent_ends, k_best_antecedent_ends))


if __name__ == '__main__':
    unittest.main()