use crate::{
    image_encoder::ImageEnvoderVit, mask_decoder::MaskDecoder, prompt_encoder::PromptEndoder,
};
use candle::Result;
use candle::Shape;
use candle::Tensor;
use candle::D;
use std::collections::HashMap;

pub struct SamInput {
    /// The image as a torch tensor in 3xHxW format, already transformed for input to the model.
    pub image: Tensor,
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
    pub original_size: (usize, usize),
    pub point_coords: Option<Tensor>,
    pub point_labels: Option<Tensor>,
    pub boxes: Option<Tensor>,
    pub mask_inputs: Option<Tensor>,
}

pub struct SamOutput {
    pub masks: Tensor,
    pub iou_predictions: Tensor,
    pub low_res_logits: Tensor,
}

pub struct Sam {
    image_encoder: ImageEnvoderVit,
    prompt_encoder: PromptEndoder,
    mask_decoder: MaskDecoder,
    mask_threshold: f64,
    pixel_mean: Tensor,
    pixel_std: Tensor,
}

impl Sam {
    pub fn forward(
        &self,
        batched_input: Vec<SamInput>,
        multimask_output: bool,
    ) -> Result<Vec<SamOutput>> {
        let input_images = Tensor::stack(
            batched_input
                .iter()
                .map(|x| self.preprocess(&x.image).expect("image"))
                .collect::<Vec<Tensor>>()
                .as_slice(),
            0,
        )?;
        let images_enbeddings = self.image_encoder.forward(&input_images)?;
        let mut output = Vec::new();
        for (image_record, curr_embedding) in batched_input.iter().zip(images_enbeddings) {
            let points = match (&image_record.point_coords, &image_record.point_labels) {
                (Some(coords), Some(labels)) => Some((coords, labels)),
                _ => None,
            };

            let (sparse_embeddings, dense_embeddings) = self.prompt_encoder.forward(
                points,
                &image_record.boxes,
                &image_record.mask_inputs,
            )?;

            let (low_res_masks, iou_predictions) = self.mask_decoder.forward(
                &curr_embedding.unsqueeze(0)?,
                &self.prompt_encoder.get_dense_pe()?,
                &sparse_embeddings,
                &dense_embeddings,
                multimask_output,
            )?;

            let (_, h, w) = image_record.image.dims3()?;
            let mut masks =
                self.postprocess_masks(&low_res_masks, (h, w), image_record.original_size)?;

            masks = masks.gt(&(&masks.ones_like()? * self.mask_threshold)?)?;

            output.push(SamOutput {
                masks,
                iou_predictions,
                low_res_logits: low_res_masks,
            });
        }
        Ok(output)
    }

    fn preprocess(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = ((x - &self.pixel_mean)? / &self.pixel_std)?;
        let (_, h, w) = x.dims3()?;
        let padh = self.image_encoder.img_size - h;
        let padw = self.image_encoder.img_size - w;
        x = x.pad_with_zeros(D::Minus1, 0, padw)?;
        x = x.pad_with_zeros(D::Minus2, 0, padh)?;
        Ok(x)
    }

    fn postprocess_masks(
        &self,
        masks: &Tensor,
        input_size: (usize, usize),
        orignal_size: (usize, usize),
    ) -> Result<Tensor> {
        let img_size = self.image_encoder.img_size;
        // TODO: This uses bilinear interpolation in the original implementation.
        let mut masks = masks.upsample_nearest2d(img_size, img_size)?;

        // remove padding
        masks = masks.narrow(D::Minus1, 0, input_size.1)?;
        masks = masks.narrow(D::Minus2, 0, input_size.0)?;

        // TODO: This uses bilinear interpolation in the original implementation.
        masks = masks.upsample_nearest2d(orignal_size.0, orignal_size.1)?;
        Ok(masks)
    }
}
