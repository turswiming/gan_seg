"""
Test script to verify KNN Distance Loss gradient propagation.

This script tests whether the KNN Distance Loss correctly computes and propagates
gradients through the point cloud tensors.
"""

import torch
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from losses.KNNDistanceLoss import KNNDistanceLoss, TruncatedKNNDistanceLoss


def test_gradient_propagation():
    """Test gradient propagation through KNN Distance Loss."""
    print("=== Testing KNN Distance Loss Gradient Propagation ===\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test point clouds with gradient tracking
    batch_size = 2
    n_points1 = 100
    n_points2 = 80
    
    # Source point cloud (requires gradient)
    source_pc = torch.randn(batch_size, n_points1, 3, device=device, requires_grad=True)
    
    # Target point cloud (requires gradient)
    target_pc = torch.randn(batch_size, n_points2, 3, device=device, requires_grad=True)
    
    print(f"Source PC shape: {source_pc.shape}, requires_grad: {source_pc.requires_grad}")
    print(f"Target PC shape: {target_pc.shape}, requires_grad: {target_pc.requires_grad}")
    print()
    
    # Test 1: Basic KNN Loss (k=1, bidirectional)
    print("Test 1: Basic KNN Loss (k=1, bidirectional)")
    print("-" * 50)
    
    loss_fn1 = KNNDistanceLoss(k=1, reduction='mean')
    loss1 = loss_fn1(source_pc, target_pc, bidirectional=True)
    
    print(f"Loss value: {loss1.item():.6f}")
    print(f"Loss requires_grad: {loss1.requires_grad}")
    print(f"Loss grad_fn: {loss1.grad_fn}")
    
    # Compute gradients
    loss1.backward()
    
    print(f"Source PC gradient shape: {source_pc.grad.shape if source_pc.grad is not None else 'None'}")
    print(f"Target PC gradient shape: {target_pc.grad.shape if target_pc.grad is not None else 'None'}")
    print(f"Source PC gradient norm: {source_pc.grad.norm().item() if source_pc.grad is not None else 'None'}")
    print(f"Target PC gradient norm: {target_pc.grad.norm().item() if target_pc.grad is not None else 'None'}")
    print()
    
    # Clear gradients for next test
    source_pc.grad = None
    target_pc.grad = None
    
    # Test 2: KNN Loss with k=5 (unidirectional)
    print("Test 2: KNN Loss with k=5 (unidirectional)")
    print("-" * 50)
    
    loss_fn2 = KNNDistanceLoss(k=5, reduction='mean')
    loss2 = loss_fn2(source_pc, target_pc, bidirectional=False)
    
    print(f"Loss value: {loss2.item():.6f}")
    print(f"Loss requires_grad: {loss2.requires_grad}")
    print(f"Loss grad_fn: {loss2.grad_fn}")
    
    # Compute gradients
    loss2.backward()
    
    print(f"Source PC gradient shape: {source_pc.grad.shape if source_pc.grad is not None else 'None'}")
    print(f"Target PC gradient shape: {target_pc.grad.shape if target_pc.grad is not None else 'None'}")
    print(f"Source PC gradient norm: {source_pc.grad.norm().item() if source_pc.grad is not None else 'None'}")
    print(f"Target PC gradient norm: {target_pc.grad.norm().item() if target_pc.grad is not None else 'None'}")
    print()
    
    # Clear gradients for next test
    source_pc.grad = None
    target_pc.grad = None
    
    # Test 3: Truncated KNN Loss
    print("Test 3: Truncated KNN Loss (k=3, threshold=1.0)")
    print("-" * 50)
    
    loss_fn3 = TruncatedKNNDistanceLoss(k=3, distance_threshold=1.0)
    loss3 = loss_fn3(source_pc, target_pc, forward_only=True)
    
    print(f"Loss value: {loss3.item():.6f}")
    print(f"Loss requires_grad: {loss3.requires_grad}")
    print(f"Loss grad_fn: {loss3.grad_fn}")
    
    # Compute gradients
    loss3.backward()
    
    print(f"Source PC gradient shape: {source_pc.grad.shape if source_pc.grad is not None else 'None'}")
    print(f"Target PC gradient shape: {target_pc.grad.shape if target_pc.grad is not None else 'None'}")
    print(f"Source PC gradient norm: {source_pc.grad.norm().item() if source_pc.grad is not None else 'None'}")
    print(f"Target PC gradient norm: {target_pc.grad.norm().item() if target_pc.grad is not None else 'None'}")
    print()


def test_gradient_flow_optimization():
    """Test gradient flow in an optimization scenario."""
    print("=== Testing Gradient Flow in Optimization ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create target point cloud (fixed)
    target_pc = torch.randn(1, 50, 3, device=device)
    
    # Create learnable source point cloud
    source_pc = torch.randn(1, 50, 3, device=device, requires_grad=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam([source_pc], lr=0.01)
    
    # Create loss function
    loss_fn = KNNDistanceLoss(k=1, reduction='mean')
    
    print("Optimizing source point cloud to match target...")
    print("Initial source PC mean:", source_pc.mean().item())
    print("Target PC mean:", target_pc.mean().item())
    
    # Optimization loop
    initial_loss = None
    for i in range(10):
        optimizer.zero_grad()
        
        loss = loss_fn(source_pc, target_pc, bidirectional=True)
        loss.backward()
        
        if i == 0:
            initial_loss = loss.item()
        
        if i % 2 == 0:
            print(f"Step {i}: Loss = {loss.item():.6f}, Grad norm = {source_pc.grad.norm().item():.6f}")
        
        optimizer.step()
    
    final_loss = loss.item()
    print(f"\nOptimization completed:")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduction: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
    
    # Check if optimization worked
    if final_loss < initial_loss:
        print("✅ Gradient flow working correctly - loss decreased during optimization")
    else:
        print("❌ Potential gradient flow issue - loss did not decrease")


def test_different_input_formats():
    """Test gradient propagation with different input formats."""
    print("=== Testing Different Input Formats ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = KNNDistanceLoss(k=1, reduction='mean')
    
    # Test 1: 2D input (single point cloud)
    print("Test 1: 2D input tensors")
    pc1_2d = torch.randn(100, 3, device=device, requires_grad=True)
    pc2_2d = torch.randn(80, 3, device=device, requires_grad=True)
    
    loss_2d = loss_fn(pc1_2d, pc2_2d, bidirectional=True)
    loss_2d.backward()
    
    print(f"2D Loss: {loss_2d.item():.6f}")
    print(f"2D PC1 gradient norm: {pc1_2d.grad.norm().item():.6f}")
    print(f"2D PC2 gradient norm: {pc2_2d.grad.norm().item():.6f}")
    print()
    
    # Test 2: 3D input (batched point clouds)
    print("Test 2: 3D input tensors (batched)")
    pc1_3d = torch.randn(2, 100, 3, device=device, requires_grad=True)
    pc2_3d = torch.randn(2, 80, 3, device=device, requires_grad=True)
    
    loss_3d = loss_fn(pc1_3d, pc2_3d, bidirectional=True)
    loss_3d.backward()
    
    print(f"3D Loss: {loss_3d.item():.6f}")
    print(f"3D PC1 gradient norm: {pc1_3d.grad.norm().item():.6f}")
    print(f"3D PC2 gradient norm: {pc2_3d.grad.norm().item():.6f}")
    print()


if __name__ == "__main__":
    # Run all tests
    test_gradient_propagation()
    test_gradient_flow_optimization()
    test_different_input_formats()
    
    print("🎉 All gradient tests completed!")


