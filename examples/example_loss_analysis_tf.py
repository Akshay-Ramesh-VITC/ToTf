"""
Example: Loss Curve Analysis with TensorFlow SmartSummary

This script demonstrates the analyze_loss_curve() method for understanding
training dynamics and identifying issues like overfitting, divergence, or plateaus.
"""

import tensorflow as tf
from tensorflow import keras
from tenf.smartsummary import SmartSummary


def create_simple_model():
    """Create a simple Sequential model"""
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model


def main():
    # Create model
    model = create_simple_model()
    
    # Create SmartSummary
    summary = SmartSummary(model, input_shape=(784,))
    
    print("\n" + "="*80)
    print("TensorFlow SmartSummary - Loss Curve Analysis Examples")
    print("="*80)
    
    # Example 1: Healthy Converging Training
    print("\n\n" + "="*80)
    print("Example 1: Healthy Converging Training")
    print("="*80)
    train_losses_good = [2.5, 2.1, 1.8, 1.6, 1.45, 1.35, 1.28, 1.24, 1.22, 1.21]
    val_losses_good = [2.6, 2.2, 1.9, 1.7, 1.55, 1.48, 1.43, 1.41, 1.40, 1.39]
    
    result1 = summary.analyze_loss_curve(train_losses_good, val_losses_good, verbose=True)
    
    # Example 2: Overfitting Scenario
    print("\n\n" + "="*80)
    print("Example 2: Overfitting Detected")
    print("="*80)
    train_losses_overfit = [2.5, 2.0, 1.6, 1.3, 1.1, 0.9, 0.75, 0.62, 0.51, 0.43]
    val_losses_overfit = [2.6, 2.1, 1.8, 1.7, 1.75, 1.85, 1.95, 2.1, 2.25, 2.4]
    
    result2 = summary.analyze_loss_curve(train_losses_overfit, val_losses_overfit, verbose=True)
    
    # Example 3: Diverging Training
    print("\n\n" + "="*80)
    print("Example 3: Diverging Training (Instability)")
    print("="*80)
    train_losses_diverge = [2.5, 2.8, 3.2, 3.9, 4.8, 6.1, 8.2, 11.5, 16.3, 23.7]
    val_losses_diverge = [2.6, 2.9, 3.4, 4.2, 5.3, 6.9, 9.1, 12.8, 18.1, 26.2]
    
    result3 = summary.analyze_loss_curve(train_losses_diverge, val_losses_diverge, verbose=True)
    
    # Example 4: Oscillating Training
    print("\n\n" + "="*80)
    print("Example 4: Oscillating Training (High Variance)")
    print("="*80)
    train_losses_oscillate = [2.5, 1.8, 2.3, 1.6, 2.1, 1.5, 2.0, 1.4, 1.9, 1.3]
    val_losses_oscillate = [2.6, 1.9, 2.4, 1.7, 2.2, 1.6, 2.1, 1.5, 2.0, 1.4]
    
    result4 = summary.analyze_loss_curve(train_losses_oscillate, val_losses_oscillate, verbose=True)
    
    # Example 5: Plateau
    print("\n\n" + "="*80)
    print("Example 5: Training Plateau")
    print("="*80)
    train_losses_plateau = [2.5, 2.0, 1.6, 1.3, 1.15, 1.12, 1.11, 1.105, 1.103, 1.102, 1.101, 1.100]
    val_losses_plateau = [2.6, 2.1, 1.7, 1.4, 1.25, 1.22, 1.21, 1.205, 1.203, 1.202, 1.201, 1.200]
    
    result5 = summary.analyze_loss_curve(train_losses_plateau, val_losses_plateau, verbose=True)
    
    # Example 6: Programmatic access to results
    print("\n\n" + "="*80)
    print("Example 6: Programmatic Access to Analysis Results")
    print("="*80)
    
    train_losses = [2.5, 2.1, 1.8, 1.6, 1.5, 1.45, 1.42, 1.41]
    result = summary.analyze_loss_curve(train_losses, verbose=False)
    
    print(f"\nStatus: {result['status']}")
    print(f"Trend: {result['trend']}")
    print(f"Stability: {result['stability']}")
    print(f"\nKey Metrics:")
    print(f"  - Improvement: {result['metrics']['improvement_percent']:.2f}%")
    print(f"  - Coefficient of Variation: {result['metrics']['coefficient_of_variation']:.2f}%")
    print(f"  - Trend Slope: {result['metrics']['trend_slope']:.6f}")
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    # Example 7: Using with actual training history
    print("\n\n" + "="*80)
    print("Example 7: Integration with Keras Training History")
    print("="*80)
    
    # Simulate training (you can replace this with actual model.fit())
    simulated_history = {
        'loss': [2.3, 1.9, 1.6, 1.4, 1.25, 1.15, 1.08, 1.04, 1.01, 0.99],
        'val_loss': [2.4, 2.0, 1.7, 1.5, 1.35, 1.28, 1.24, 1.22, 1.21, 1.20]
    }
    
    print("\nAnalyzing training history from model.fit()...")
    result = summary.analyze_loss_curve(
        train_losses=simulated_history['loss'],
        val_losses=simulated_history['val_loss'],
        verbose=True
    )
    
    print("\n\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
