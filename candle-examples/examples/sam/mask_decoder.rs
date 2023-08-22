use candle::{Result, Tensor};

pub struct MaskDecoder {}

impl MaskDecoder {
    pub fn forward(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        todo!()
    }
}
