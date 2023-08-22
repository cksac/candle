use candle::{Result, Tensor};
pub struct PromptEndoder {}

impl PromptEndoder {
    pub fn forward(
        &self,
        points: Option<(&Tensor, &Tensor)>,
        boxes: &Option<Tensor>,
        mask_inputs: &Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        todo!()
    }

    pub fn get_dense_pe(&self) -> Result<Tensor> {
        todo!()
    }
}
