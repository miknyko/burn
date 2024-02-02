use burn_tensor::backend::Backend;

use crate as burn;

use super::LrScheduler;
use crate::{config::Config, LearningRate};

/// Configure a linear warm up [warmup](WarmUpSchedulerConfig) learning rate scheduler.
#[derive(Config)]
pub struct WarmUpSchedulerConfig {
    /// The initial learning rate for the schedule after the warmup (so this will be the final learning rate at the end of the warmup)
    init_lr: LearningRate,
    /// Number of warming up steps
    #[config(default = 5)]
    num_warmup_steps: usize,
    /// The power to use for the polynomial warmup (defaults is a linear warmup)
    #[config(default = 1.0)]
    power: f64,
}

/// Applies a learning rate warmup schedule to a given rate.
#[derive(Clone, Debug)]
pub struct WarmUpScheduler {
    init_lr: LearningRate,
    num_warmup_steps: f64,
    power: f64,
    step: f64,
}

impl WarmUpSchedulerConfig {
    /// Initialize a new [warmup](WarmUpSchedulerConfig) learning rate scheduler.
    pub fn init(&self) -> WarmUpScheduler {
        WarmUpScheduler {
            init_lr: self.init_lr,
            num_warmup_steps: self.num_warmup_steps as f64,
            power: self.power,
            step: 0.0,
        }
    }
}

impl<B: Backend> LrScheduler<B> for WarmUpScheduler {
    type Record = usize;

    fn step(&mut self) -> LearningRate {
        self.step += 1.0;

        if self.step < self.num_warmup_steps {
            let factor = self.step / self.num_warmup_steps.powf(self.power);
            self.init_lr * factor
        } else {
            self.init_lr
        }
    }

    fn to_record(&self) -> Self::Record {
        self.step as usize
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.step = record as f64;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Data, Float, Tensor};
    use crate::TestBackend;

    #[test]
    fn test_warm_up_scheduler_step() {
        let config = WarmUpSchedulerConfig {
            init_lr: 0.1,
            num_warmup_steps: 5,
            power: 1.0,
        };
        let mut scheduler = config.init();
        let device = Default::default();
        let expected_tensor = Tensor::<TestBackend, 1, Float>::from_floats(
            [0.02, 0.04, 0.06, 0.08, 0.1, 0.1, 0.1],
            &device,
        );


        let mut actual: [f32; 7] = [0.0; 7];
        for i in 0..7 {
            let lr = LrScheduler::<TestBackend>::step(&mut scheduler);
            actual[i] = lr as f32;
        }
        let actual_tensor = Tensor::<TestBackend, 1, Float>::from_data(actual, &device);

        actual_tensor
            .to_data()
            .assert_approx_eq(&expected_tensor.to_data(), 5);
    }
}
