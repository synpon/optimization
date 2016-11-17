import numpy as np
import tensorflow as tf
import accum_trainer

class AccumTrainerTest(tf.test.TestCase):
  def testAccum(self):
    with self.test_session():
      var0 = tf.Variable([1.0, 2.0])
      trainer = accum_trainer.AccumTrainer()
      
      cost = tf.square(var0)
      
      trainer.prepare_minimize(cost, [var0])
      
      accumulate_grad = trainer.accumulate_gradients()
      reset = trainer.reset_gradients()
      
      tf.initialize_all_variables().run()

      # grad execution of addition
      accumulate_grad.run()
      
      # accumulate_grad even if the var0 does not change its contents
      self.assertAllClose([1.0, 2.0], var0.eval())
      
      accum_grads = trainer._accum_grad_list
      accum_grad0 = accum_grads[0]

      # Confirming that the grad is added to accum_grad
      self.assertAllClose([2.0, 4.0], accum_grad0.eval())

      # Run again the addition of grad
      accumulate_grad.run()

      # grad is sure to have been further added to accum_grad
      self.assertAllClose([4.0, 8.0], accum_grad0.eval())

      # reset the execution
      reset.run()

      # Confirmed that the accum_grad is zero
      self.assertAllClose([0.0, 0.0], accum_grad0.eval())

  # TODO: gradient clipping test
      

if __name__ == "__main__":
  tf.test.main()
