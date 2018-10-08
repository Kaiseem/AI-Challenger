from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
#from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_ops
import tensorflow as tf


class SwatsOptimizer(optimizer.Optimizer):
    '''
    http://cn.arxiv.org/pdf/1712.07628

    '''
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, epsilon=1e-8, name="Adam"):
        super(SwatsOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._beta1_power = None
        self._beta2_power = None
        self.SGD = tf.constant(False)
        self.step=0
        self.v_t = 0
        self.lambda_t=0

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        first_var = min(var_list, key=lambda x: x.name)
        with ops.colocate_with(first_var):
            self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
            self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m1", self._name)
            self._zeros_slot(v, "v1", self._name)
    def SGD(self,grad,var):
        self.v_t=self._beta1*self.v_t+grad
        var_update=state_ops.assign_sub(var, (1-self._beta1)*self.A*self.v_t)
        return control_flow_ops.group(*[var_update])
    def ojbk(self):
        self.SGD = True
        self.v_t = 0
        self.A = self.lambda_t / (1 - self._beta2)
        return None
    def Adam(self,grad,var):
        eps = 1e-8
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        v = self.get_slot(var, "v1")
        v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad**2)
        m = self.get_slot(var, "m1")
        m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)
        alpha_t =  tf.sqrt(1 - beta2_power) / (1 - beta1_power)
        g_t =  (m_t*alpha_t) / (tf.sqrt(v_t) + eps)
        var_update = state_ops.assign_sub(var, lr_t * g_t)
        gamma_t   =tf.cond(tf.equal( tf.matmul(g_t,grad, transpose_a=True),0),self.lambda_t,-tf.matmul(g_t,g_t, transpose_a=True)/tf.matmul(g_t,grad, transpose_a=True))
        self.lambda_t=beta2_t*self.lambda_t+(1-beta2_t)*gamma_t
        a=tf.cond(tf.abs(self.lambda_t/(1-beta2_t)-gamma_t)<10e-9,lambda :self.ojbk,lambda :None)
        return control_flow_ops.group(*[var_update, v_t, m_t])

    def _apply_dense(self, grad, var):
        self.step+=1
        return tf.cond(self.SGD,lambda :self.SGD(grad,var),lambda:self.Adam(grad,var))

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")