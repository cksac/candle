use candle::{Result, Tensor};

pub struct ImageEnvoderVit {
    pub(crate) img_size: usize,
}

impl ImageEnvoderVit {
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        todo!()
    }
}
