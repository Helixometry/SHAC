"""Unit tests for GROOT model."""

import unittest
import torch
from shac.model import GROOT


class TestGROOT(unittest.TestCase):
    """Test GROOT model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = GROOT(
            spectral_dim=13,
            ssl_dim=768,
            hidden_dim=256,
        )
        self.batch_size = 4
        self.seq_len = 100

    def test_model_initialization(self):
        """Test model initializes."""
        self.assertIsNotNone(self.model)

    def test_forward_pass(self):
        """Test forward pass."""
        spectral = torch.randn(self.batch_size, self.seq_len, 13)
        ssl = torch.randn(self.batch_size, self.seq_len, 768)

        outputs = self.model(spectral, ssl)

        self.assertIn('logits', outputs)
        self.assertIn('real_prob', outputs)
        self.assertIn('synthetic_prob', outputs)
        self.assertEqual(outputs['logits'].shape[0], self.batch_size)


if __name__ == '__main__':
    unittest.main()
