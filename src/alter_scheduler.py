import torch
class AlterScheduler:
    def __init__(self, alter_config):
        self.flow_list = alter_config.flow
        self.mask_list = alter_config.mask

        self.flow_stepsum = sum([ item[0] for item in self.flow_list ])
        self.mask_stepsum = sum([ item[0] for item in self.mask_list ])
        self.iter = 0

    def step(self):
        self.iter += 1

    def flow_train(self):
        """
        :return: isflowtrain: bool
        """
        step = self.iter % self.flow_stepsum
        for step_num, train  in self.flow_list:
            # print("step_num", step_num, "train", train)
            if step < step_num:
                return train
            step -= step_num
        return self.flow_list[-1][1]
    
    def mask_train(self):
        """
        :return: ismasktrain: bool
        """
        step = self.iter % self.mask_stepsum
        for step_num, train in self.mask_list:
            # print("step_num", step_num, "train", train)
            if step < step_num:
                return train
            step -= step_num
        return self.mask_list[-1][1]
    #prepare for save
    def state_dict(self):
        return {
            "iter": self.iter,
        }
    def load_state_dict(self, state_dict):
        self.iter = state_dict["iter"]


class SceneFlowSmoothnessScheduler:
    def __init__(self, scene_flow_smoothness_config):
        self.begin_iter = scene_flow_smoothness_config.begin_iter
        self.end_iter = scene_flow_smoothness_config.end_iter
        self.begin_value = scene_flow_smoothness_config.begin_value
        self.end_value = scene_flow_smoothness_config.end_value
        if self.begin_iter >= self.end_iter:
            raise ValueError("begin_iter must be less than end_iter")
    def __call__(self, iter):
        if iter < self.begin_iter:
            return self.begin_value
        elif iter < self.end_iter:
            return self.end_value
        else:
            return (iter-self.begin_iter)/(self.end_iter-self.begin_iter)*(self.end_value-self.begin_value)+self.begin_value


class MaskLRScheduler:
    """学习率调度器，用于mask model的学习率调整"""
    def __init__(self, optimizer, scheduler_config):
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.scheduler = None
        
        # 根据配置创建相应的scheduler
        if hasattr(scheduler_config, 'type'):
            scheduler_type = scheduler_config.type
        else:
            scheduler_type = 'step'  # 默认使用StepLR
            
        if scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        elif scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_config.gamma
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.mode,
                factor=scheduler_config.factor,
                patience=scheduler_config.patience,
                threshold=scheduler_config.threshold
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def step(self, metrics=None):
        """执行scheduler step"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metrics is not None:
                    self.scheduler.step(metrics)
                else:
                    self.scheduler.step()
            else:
                self.scheduler.step()
    
    def get_last_lr(self):
        """获取当前学习率"""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """保存scheduler状态"""
        if self.scheduler is not None:
            return self.scheduler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict):
        """加载scheduler状态"""
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict)