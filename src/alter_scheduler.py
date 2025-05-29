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
        for flow, step_num in self.flow_list:
            if step < step_num:
                return flow
            step -= step_num
        return self.flow_list[-1][1]
    
    def mask_train(self):
        """
        :return: ismasktrain: bool
        """
        step = self.iter % self.mask_stepsum
        for mask, step_num in self.mask_list:
            if step < step_num:
                return mask
            step -= step_num
        return self.mask_list[-1][1]
