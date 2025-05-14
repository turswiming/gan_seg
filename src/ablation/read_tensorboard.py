from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_data(log_dir, tag_name):
    """
    从TensorBoard日志文件中读取特定标签的数据
    
    Args:
        log_dir: TensorBoard日志目录路径
        tag_name: 要读取的数据标签名称
    
    Returns:
        values: 值列表
        steps: 对应的步骤列表
    """
    # 创建事件累加器
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0表示加载所有数据
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        }
    )
    
    # 加载日志数据
    ea.Reload()
    
    # 检查标签是否存在
    available_tags = ea.Tags()['scalars']
    if tag_name not in available_tags:
        print(f"Tag '{tag_name}' not found. Available tags: {available_tags}")
        return [], []
        
    # 读取标量数据
    scalar_events = ea.Scalars(tag_name)
    values = [event.value for event in scalar_events]
    steps = [event.step for event in scalar_events]
    
    return steps,values