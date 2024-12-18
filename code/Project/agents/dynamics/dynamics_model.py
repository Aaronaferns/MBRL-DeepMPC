from agents.dynamics.model_utils import *
import torch as th

class DynamicsModel:
    def __init__(self,
                 nS,
                 nA,
                 pred_rew=False, 
                 data_buffer_size = 50000, 
                 val_split=0.2
                 ):
        self._input_size = nS+nA
        self._output_size = nS
        if pred_rew: self._output_size +=1
        self._pred_rew = pred_rew
        self._buf = DataBuffer(nS,nA,self._pred_rew,data_buffer_size)
        self._val_buf = DataBuffer(nS,nA,self._pred_rew,data_buffer_size)

    
    @property
    def data_count(self):
        return self._buf.buffer_counter

    #Abstract methods child classes have to implement
    def train(self, num_epochs=1000):
        raise NotImplementedError

    def predict(self, curr_batch):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError

    def save_model(self, save_dir):
        raise NotImplementedError

    def load_model(self, save_dir):
        raise NotImplementedError

    def set_model_to(self, model_to_copy):
        raise NotImplementedError


    def push_data(self, curr_state, act, next_state, rew=None):
        self._buf.add(curr_state, act, next_state, rew)

    def push_val_data(self, curr_state, act, next_state, rew=None):
        self._val_buf.add(curr_state, act, next_state, rew)

    #batch for training the dynamics model
    def get_data(self, is_delta=True):
        batch = self._buf.get_batch()
        val_batch = self._val_buf.get_batch()
        
        if self._pred_rew:
            s, a, r, s_ = batch
            val_s, val_a, val_r, val_s_ = val_batch
            if is_delta:
                s_ = self.getStateDelta(s,s_)
                val_s_ = self.getStateDelta(val_s, val_s_)
            output = np.concatenate([s,r], axis=1)
            val_output = np.concatenate([val_s,val_a],axis=1)

        #we're using delta s for the model training
        else: 
            s,a,s_ = batch
            val_s, val_a, val_s_ = val_batch
            if is_delta:
                s_ = self.getStateDelta(s,s_)
                val_s_ = self.getStateDelta(val_s, val_s_)
            output = s_
            val_output = val_s_
            
        input = np.concatenate([s,a], axis=1)
        val_input = np.concatenate([val_s,val_a],axis=1)
        
        if self._pred_rew: return th.tensor(input), th.tensor(output), th.tensor(val_input), th.tensor(val_output), th.tensor(r), th.tensor(val_r)
        return th.tensor(input), th.tensor(output), th.tensor(val_input), th.tensor(val_output)
    
    #Compute the deltas
    def getStateDelta(self,batch_s,batch_s_):
        return batch_s_-batch_s
    
    #get the next state depending on the predicted deltas
    def get_new_states(self, state_batch, delta_batch):
        next_state_batch = state_batch[:,:self._output_size] + delta_batch[:,:self._output_size] 
        return th.concat((next_state_batch[:,:self._output_size], delta_batch[:,self._output_size:]), axis=1)