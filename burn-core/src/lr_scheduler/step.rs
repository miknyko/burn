use burn_tensor::backend::Backend;

use crate as burn;

use super::LrScheduler;
use crate::{config::Config, LearningRate};

/// Configuration to create a [step](StepLrScheduler) learning rate scheduler.
#[derive(Config)]
pub struct StepLrSchedulerConfig {
    /// The initial learining
    init_lr: LearningRate,
    /// Period of learning rate decay
    #[config(default = 10)]
    step_size: i64,
    /// Multiplicative factor of learning rate decay
    #[config(default = 0.1)]
    gamma: f64,
}

/// Decays the learning rate of each parameter group by gamma every step_size epochs.
#[derive(Clone, Debug)]
pub struct StepLrScheduler {
    init_lr: LearningRate,
    step_size: i64,
    gamma: f64,
    step: f64,
}

impl StepLrSchedulerConfig {
    /// Initialize a new [step](StepLrScheduler) learning rate scheduler.
    pub fn init(&self) -> StepLrScheduler {
        StepLrScheduler {
            init_lr: self.init_lr,
            step_size: self.step_size,
            gamma: self.gamma,
            step: 0.0,
        }
    }
}

impl<B: Backend> LrScheduler<B> for StepLrScheduler {
    type Record = usize;

    fn step(&mut self) -> LearningRate {
        self.step += 1.0;
        let factor = self.gamma.powf((self.step as i64 / self.step_size) as f64);
        self.init_lr * factor
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

    use crate::TestBackend;

    use super::*;

    #[test]
    fn test_step_lr_scheduler() {
        let init_lr = 10.0;
        let step = 2;
        let gamma = 0.9;
        let mut scheduler = StepLrSchedulerConfig::new(init_lr)
            .with_step_size(step)
            .with_gamma(gamma)
            .init();

        for epoch in 1..10 {
            let lr = LrScheduler::<TestBackend>::step(&mut scheduler);
            let expected = init_lr * gamma.powf((epoch / step) as f64);
            assert!(
                lr == expected,
                "Learning rate should decrease as scheduled."
            );
        }
    }
}
