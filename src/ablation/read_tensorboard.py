"""
TensorBoard log reader utility.

This module provides functionality to read scalar data from TensorBoard log files.
"""

from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_data(log_dir, tag_name):
    """
    Read data for a specific tag from TensorBoard log files.
    
    Args:
        log_dir (str): Path to the TensorBoard log directory
        tag_name (str): Name of the data tag to read
    
    Returns:
        tuple: A tuple containing:
            - steps (list): List of training steps
            - values (list): List of corresponding values
    """
    # Create event accumulator
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 means load all data
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        }
    )
    
    # Load log data
    ea.Reload()
    
    # Check if tag exists
    available_tags = ea.Tags()['scalars']
    if tag_name not in available_tags:
        print(f"Tag '{tag_name}' not found. Available tags: {available_tags}")
        return [], []
        
    # Read scalar data
    scalar_events = ea.Scalars(tag_name)
    values = [event.value for event in scalar_events]
    steps = [event.step for event in scalar_events]
    
    return steps, values