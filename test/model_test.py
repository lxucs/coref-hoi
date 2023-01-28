import unittest
import sys
sys.path.append("..")
import model
import torch

class TestModel(unittest.TestCase):

    def test_get_k_best_predicted_antecedents(self):
        antecedent_idx = [[1, 2, 3, 4], [5, 6, 7, 8]]
        antecedent_scores = [[0, 20, 50, 30, 40], [0, 60, 30, 80, 40]]
        k = 3
        expected_k_best_antecedent_scores = torch.tensor([[50, 40, 30], [80, 60, 40]])
        expected_k_best_antecedent_idx = torch.tensor([[2, 4, 3], [7, 5, 8]])
        
        k_best_antecedent_idx, k_best_antecedent_scores = model.CorefModel.get_k_best_predicted_antecedents(antecedent_idx, antecedent_scores, k)
        self.assertTrue(torch.equal(expected_k_best_antecedent_scores, k_best_antecedent_scores))
        self.assertTrue(torch.equal(expected_k_best_antecedent_idx, k_best_antecedent_idx))

    def test_get_k_best_predicted_antecedents_dummy(self):
        antecedent_idx = [[1, 2, 3, 4], [5, 6, 7, 8]]
        antecedent_scores = [[0, 20, 50, 30, 40], [0, 60, -10, 80, -20]]
        k = 3
        expected_k_best_antecedent_scores = torch.tensor([[50, 40, 30], [80, 60, 0]])
        expected_k_best_antecedent_idx = torch.tensor([[2, 4, 3], [7, 5, -1]])

        k_best_antecedent_idx, k_best_antecedent_scores = model.CorefModel.get_k_best_predicted_antecedents(antecedent_idx, antecedent_scores, k)
        self.assertTrue(torch.equal(expected_k_best_antecedent_scores, k_best_antecedent_scores))
        self.assertTrue(torch.equal(expected_k_best_antecedent_idx, k_best_antecedent_idx))


if __name__ == '__main__':
    unittest.main()