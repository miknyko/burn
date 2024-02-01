/// Constant learning rate scheduler
pub mod constant;

/// Noam Learning rate schedule
pub mod noam;
/// Step Learning rate schedule
pub mod step;

mod base;

pub use base::*;
