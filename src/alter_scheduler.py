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