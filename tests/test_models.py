import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import torch
from models.hstf import build_hstf
from models.nftp import build_nftp

class TestPPGPT(unittest.TestCase):

    def setUp(self):
        self.B, self.T, self.V = (4, 512, 32)
        self.enc = build_hstf({'dim': 32, 'layers': 5, 'patches': 16})
        self.sig = torch.randn(self.B, self.T)
        self.vis = torch.rand(self.B, self.V, self.V, 3)

    def test_hstf_token_shape(self):
        Z = self.enc(self.sig, self.vis)
        self.assertEqual(Z.shape, (self.B, 5, 2 * 32))

    def test_hstf_projection(self):
        Z, p = self.enc(self.sig, self.vis, return_final=True)
        self.assertEqual(p.shape, (self.B, 32))

    def test_nftp_loss_and_shape(self):
        dec = build_nftp(self.enc.token_dim(), {'d_model': 48, 'depth': 2, 'max_len': 8})
        Z = self.enc(self.sig, self.vis)
        pred, aux = dec(Z)
        self.assertEqual(pred.shape, Z.shape)
        loss, logs = dec.nftp_loss(Z)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('nftp_mse', logs)
if __name__ == '__main__':
    unittest.main()
