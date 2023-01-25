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
        
        k_best_antecedents, k_best_antecedent_scores = model.CorefModel.get_k_best_predicted_antecedents(antecedent_idx, antecedent_scores, k)

        expected_k_best_antecedent_scores = torch.tensor([[50, 40, 30], [80, 60, 40]])
        expected_k_best_antecedents = torch.tensor([[2, 4, 3], [7, 5, 8]])
        
        self.assertTrue(torch.equal(expected_k_best_antecedent_scores, k_best_antecedent_scores))
        self.assertTrue(torch.equal(expected_k_best_antecedents, k_best_antecedents))


if __name__ == '__main__':
    unittest.main()