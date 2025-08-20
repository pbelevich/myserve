import unittest
import torch

from myserve.core.collate import pad_past, split_past, pad_sequences


class TestCollate(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cpu')
        self.dtype = torch.float32
        
        # Create sample past data for testing
        self.create_sample_pasts()
    
    def create_sample_pasts(self):
        """Create sample past data with different sequence lengths."""
        # Create pasts with different sequence lengths
        # Past structure: List[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
        # Each layer has (K, V) where K, V are [1, H, T, D]
        
        # Past 1: length 3
        k1 = torch.randn((1, 4, 3, 8), device=self.device, dtype=self.dtype)  # [1, H=4, T=3, D=8]
        v1 = torch.randn((1, 4, 3, 8), device=self.device, dtype=self.dtype)
        past1 = ((k1, v1),)
        
        # Past 2: length 5
        k2 = torch.randn((1, 4, 5, 8), device=self.device, dtype=self.dtype)  # [1, H=4, T=5, D=8]
        v2 = torch.randn((1, 4, 5, 8), device=self.device, dtype=self.dtype)
        past2 = ((k2, v2),)
        
        # Past 3: length 2
        k3 = torch.randn((1, 4, 2, 8), device=self.device, dtype=self.dtype)  # [1, H=4, T=2, D=8]
        v3 = torch.randn((1, 4, 2, 8), device=self.device, dtype=self.dtype)
        past3 = ((k3, v3),)
        
        self.pasts = [past1, past2, past3]
        self.expected_max_length = 5
    
    def test_pad_past_basic(self):
        """Test basic padding functionality."""
        padded, lengths, attn_mask, pos_ids = pad_past(self.pasts)
        
        # Check return types
        self.assertIsInstance(padded, tuple)
        self.assertIsInstance(lengths, torch.Tensor)
        self.assertIsInstance(attn_mask, torch.Tensor)
        self.assertIsInstance(pos_ids, torch.Tensor)
        
        # Check lengths
        expected_lengths = torch.tensor([3, 5, 2], dtype=torch.long)
        torch.testing.assert_close(lengths, expected_lengths)
        
        # Check padded past structure
        self.assertEqual(len(padded), 1)  # One layer
        K, V = padded[0]
        self.assertEqual(K.shape, (3, 4, 5, 8))  # [B=3, H=4, T_max=5, D=8]
        self.assertEqual(V.shape, (3, 4, 5, 8))
        
        # Check that original data is preserved
        torch.testing.assert_close(K[0, :, -3:, :], self.pasts[0][0][0][0,:])  # First past, first 3 tokens
        torch.testing.assert_close(V[0, :, -3:, :], self.pasts[0][0][1][0,:])
        
        # Check that padding is zeros
        torch.testing.assert_close(K[0, :, :2, :], torch.zeros(4, 2, 8, device=self.device, dtype=self.dtype))
        torch.testing.assert_close(V[0, :, :2, :], torch.zeros(4, 2, 8, device=self.device, dtype=self.dtype))
    
    def test_pad_past_attention_mask(self):
        """Test attention mask generation."""
        _, lengths, attn_mask, _ = pad_past(self.pasts)
        
        # Check attention mask shape
        self.assertEqual(attn_mask.shape, (3, 6))  # [B=3, T_max+1=6]
        
        # Check attention mask values
        expected_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1],  # Past 1: length 3, so mask [0:4] = 1
            [1, 1, 1, 1, 1, 1],  # Past 2: length 5, so mask [0:6] = 1
            [0, 0, 0, 1, 1, 1],  # Past 3: length 2, so mask [0:3] = 1
        ], dtype=torch.long)

        torch.testing.assert_close(attn_mask, expected_mask)
    
    def test_pad_past_position_ids(self):
        """Test position IDs generation."""
        _, lengths, _, pos_ids = pad_past(self.pasts)
        
        # Check position IDs shape
        self.assertEqual(pos_ids.shape, (3, 1))  # [B=3, 1]
        
        # Check position IDs values (should be the lengths)
        expected_pos_ids = torch.tensor([[3], [5], [2]], dtype=torch.long)
        torch.testing.assert_close(pos_ids, expected_pos_ids)
    
    def test_pad_past_single_past(self):
        """Test padding with a single past (no padding needed)."""
        single_past = [self.pasts[0]]  # Just past 1
        padded, lengths, attn_mask, pos_ids = pad_past(single_past)
        
        # Check that no padding was applied
        K, V = padded[0]
        self.assertEqual(K.shape, (1, 4, 3, 8))  # Original shape preserved
        self.assertEqual(V.shape, (1, 4, 3, 8))
        
        # Check lengths
        torch.testing.assert_close(lengths, torch.tensor([3], dtype=torch.long))
        
        # Check attention mask
        torch.testing.assert_close(attn_mask, torch.tensor([[1, 1, 1, 1]], dtype=torch.long))
    
    def test_pad_past_multiple_layers(self):
        """Test padding with multiple layers."""
        # Create pasts with 2 layers
        k1_l1 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        v1_l1 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        k1_l2 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        v1_l2 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        past1_2layers = ((k1_l1, v1_l1), (k1_l2, v1_l2))
        
        k2_l1 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        v2_l1 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        k2_l2 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        v2_l2 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        past2_2layers = ((k2_l1, v2_l1), (k2_l2, v2_l2))
        
        pasts_2layers = [past1_2layers, past2_2layers]
        
        padded, lengths, attn_mask, pos_ids = pad_past(pasts_2layers)
        
        # Check that both layers are padded
        self.assertEqual(len(padded), 2)
        
        # Check first layer
        K1, V1 = padded[0]
        self.assertEqual(K1.shape, (2, 4, 5, 8))
        self.assertEqual(V1.shape, (2, 4, 5, 8))
        
        # Check second layer
        K2, V2 = padded[1]
        self.assertEqual(K2.shape, (2, 4, 5, 8))
        self.assertEqual(V2.shape, (2, 4, 5, 8))
    
    def test_split_past_basic(self):
        """Test basic splitting functionality."""
        # First pad the pasts
        padded, lengths, _, _ = pad_past(self.pasts)
        
        # Then split them back
        split_pasts = split_past(padded, lengths)
        
        # Check that we get the same number of pasts back
        self.assertEqual(len(split_pasts), 3)
        
        # Check that each split past has the correct structure
        for i, split in enumerate(split_pasts):
            self.assertEqual(len(split), 1)  # One layer
            K, V = split[0]
            expected_length = lengths[i].item()

            # Check shapes
            self.assertEqual(K.shape, (1, 4, expected_length, 8))
            self.assertEqual(V.shape, (1, 4, expected_length, 8))
            
            # Check that data is contiguous
            self.assertTrue(K.is_contiguous())
            self.assertTrue(V.is_contiguous())
    
    def test_split_past_data_integrity(self):
        """Test that splitting preserves the original data."""
        padded, lengths, _, _ = pad_past(self.pasts)
        split_pasts = split_past(padded, lengths)
        
        # Check that the first past (length 3) has the correct data
        K, V = split_pasts[0][0]
        self.assertEqual(K.shape, (1, 4, 3, 8))
        
        # The data should match the original padded data for the first batch
        K_padded, V_padded = padded[0]
        torch.testing.assert_close(K, K_padded[0:1, :, -3:, :])
        torch.testing.assert_close(V, V_padded[0:1, :, -3:, :])
    
    def test_split_past_multiple_layers(self):
        """Test splitting with multiple layers."""
        # Create pasts with 2 layers
        k1_l1 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        v1_l1 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        k1_l2 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        v1_l2 = torch.randn(1, 4, 3, 8, device=self.device, dtype=self.dtype)
        past1_2layers = ((k1_l1, v1_l1), (k1_l2, v1_l2))
        
        k2_l1 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        v2_l1 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        k2_l2 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        v2_l2 = torch.randn(1, 4, 5, 8, device=self.device, dtype=self.dtype)
        past2_2layers = ((k2_l1, v2_l1), (k2_l2, v2_l2))
        
        pasts_2layers = [past1_2layers, past2_2layers]
        
        padded, lengths, _, _ = pad_past(pasts_2layers)
        split_pasts = split_past(padded, lengths)
        
        # Check that each split past has 2 layers
        for split in split_pasts:
            self.assertEqual(len(split), 2)
            
            # Check both layers
            for layer_idx in range(2):
                K, V = split[layer_idx]
                self.assertEqual(K.shape[0], 1)  # Batch size 1
                self.assertEqual(V.shape[0], 1)
    
    def test_split_past_edge_cases(self):
        """Test edge cases for splitting."""
        # Test with single past
        single_past = [self.pasts[0]]
        padded, lengths, _, _ = pad_past(single_past)
        split_pasts = split_past(padded, lengths)
        
        self.assertEqual(len(split_pasts), 1)
        K, V = split_pasts[0][0]
        self.assertEqual(K.shape, (1, 4, 3, 8)) 
        
        # Test with empty lengths tensor
        empty_lengths = torch.tensor([], dtype=torch.long)
        empty_split = split_past(padded, empty_lengths)
        self.assertEqual(len(empty_split), 0)
    
    def test_pad_past_device_dtype_preservation(self):
        """Test that device and dtype are preserved during padding."""
        # Create pasts on CPU with float32
        pasts_cpu = self.pasts
        
        padded, lengths, attn_mask, pos_ids = pad_past(pasts_cpu)
        
        # Check that all tensors have the same device and dtype
        K, V = padded[0]
        self.assertEqual(K.device, self.device)
        self.assertEqual(K.dtype, self.dtype)
        self.assertEqual(V.device, self.device)
        self.assertEqual(V.dtype, self.dtype)
        self.assertEqual(attn_mask.device, self.device)
        self.assertEqual(pos_ids.device, self.device)
    
    def test_pad_past_roundtrip(self):
        """Test that pad_past followed by split_past gives back the original data."""
        padded, lengths, _, _ = pad_past(self.pasts)
        split_pasts = split_past(padded, lengths)
        
        # Check that we get the same number of pasts back
        self.assertEqual(len(split_pasts), len(self.pasts))
        
        # Check that each past has the correct number of layers
        for i, split in enumerate(split_pasts):
            self.assertEqual(len(split), len(self.pasts[i]))
    
    def test_pad_past_different_shapes(self):
        """Test padding with pasts that have different hidden dimensions."""
        # Create pasts with different hidden dimensions
        k1 = torch.randn(1, 2, 3, 4, device=self.device, dtype=self.dtype)  # H=2, D=4
        v1 = torch.randn(1, 2, 3, 4, device=self.device, dtype=self.dtype)
        past1 = ((k1, v1),)
        
        k2 = torch.randn(1, 2, 5, 4, device=self.device, dtype=self.dtype)  # H=2, D=4
        v2 = torch.randn(1, 2, 5, 4, device=self.device, dtype=self.dtype)
        past2 = ((k2, v2),)
        
        pasts_diff_shapes = [past1, past2]
        
        padded, lengths, attn_mask, pos_ids = pad_past(pasts_diff_shapes)
        
        # Check that padding worked correctly
        K, V = padded[0]
        self.assertEqual(K.shape, (2, 2, 5, 4))  # [B=2, H=2, T_max=5, D=4]
        self.assertEqual(V.shape, (2, 2, 5, 4))
        
        # Check lengths
        expected_lengths = torch.tensor([3, 5], dtype=torch.long)
        torch.testing.assert_close(lengths, expected_lengths)

    def test_pad_sequences(self):
        input_ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
        inputs_ids, attn_mask, position_ids, lengths = pad_sequences(input_ids, pad_token_id=0)
        torch.testing.assert_close(inputs_ids, torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5], [6, 7, 8, 9]]), msg=f"{inputs_ids=}")
        torch.testing.assert_close(attn_mask, torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]), msg=f"{attn_mask=}")
        torch.testing.assert_close(position_ids, torch.tensor([[0, 0, 1, 2], [0, 0, 0, 1], [0, 1, 2, 3]]), msg=f"{position_ids=}")
        torch.testing.assert_close(lengths, torch.tensor([3, 2, 4], dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
