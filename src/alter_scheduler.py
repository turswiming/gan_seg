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
