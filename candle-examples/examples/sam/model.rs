use crate::{ImageEnvoderVit, MaskDecoder, PromptEndoder};
use candle::Result;
use candle::Shape;
use candle::Tensor;

pub struct SamInput {
    /// The image as a torch tensor in 3xHxW format, already transformed for input to the model.
    image: Tensor,
    //     'image': The image as a torch tensor in 3xHxW format,
    //     already transformed for input to the model.
    //   'original_size': (tuple(int, int)) The original size of
    //     the image before transformation, as (H, W).
    //   'point_coords': (torch.Tensor) Batched point prompts for
    //     this image, with shape BxNx2. Already transformed to the
    //     input frame of the model.
    //   'point_labels': (torch.Tensor) Batched labels for point prompts,
    //     with shape BxN.
    //   'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
    //     Already transformed to the input frame of the model.
    //   'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
    //     in the form Bx1xHxW.
    original_size: (usize, usize),
    point_coords: Option<Tensor>,
    point_labels: Option<Tensor>,
    boxes: Option<Tensor>,
    mask_inputs: Option<Tensor>,
}

pub struct Sam {
    image_encoder: ImageEnvoderVit,
    prompt_encoder: PromptEndoder,
    mask_decoder: MaskDecoder,
    mask_threshold: f32,
}

impl Sam {
    pub fn forward(
        &self,
        batched_input: Vec<SamInput>,
        multimask_output: bool,
    ) -> Vec<HashMap<String, Tensor>> {
        input_images = Tensor::stack(batched_input.map(|x| self.preprocess(x)), 0)?;
        images_enbeddings = self.image_encoder.forward(input_images)?;
        let mut output = Vec::new();

        for (image_record, curr_embedding) in batched_input.zip(images_enbeddings) {
            let points = match (batched_input.point_coords, batched_input.point_labels) {
                (Some(coords), Some(labels)) => Some((coords, labels)),
                _ => None,
            };

            let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                points,
                image_record.boxes,
                image_record.mask_inputs,
            )?;

            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                curr_embedding.unsqueeze(0)?,
                self.prompt_encoder.get_dense_pe(),
                sparse_embeddings,
                dense_embeddings,
                multimask_output,
            )?;

            let mut masks = self.postprocess_masks(
                low_res_masks,
                (0, 0), //TODO: input_size=image_record["image"].shape[-2:],
                image_record.original_size,
            )?;

            masks = masks.gt((&masks.ones_like() * self.mask_threshold)?)?;
            let mut r = HashMap::new();
            r.insert("masks", masks);
            r.insert("iou_predictions", iou_predictions);
            r.insert("low_res_logits", low_res_masks);
            output.push(r);
        }
        output
    }

    fn preprocess(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }

    fn postprocess_masks(
        &self,
        masks: &Tensor,
        input_size: (usize, usize),
        orignal_size: (usize, usize),
    ) -> Result<Tensor> {
        todo!()
    }
}
