use candle::{Result, Tensor};
use candle_nn::{conv2d, Conv2d, LayerNorm};
use candle_nn::{Module, VarBuilder};

pub struct ImageEnvoderVit {
    pub img_size: usize,
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    neck: Neck,
}

impl ImageEnvoderVit {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(x)?;
        // if self.pos_embed is not None:
        //     x = x + self.pos_embed
        for blk in self.blocks.iter() {
            x = blk.forward(&x)?
        }

        x = x.permute((0, 3, 1, 2))?;
        x = self.neck.forward(&x)?;
        Ok(x)
    }
}

pub struct Block {}
impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct Neck {
    proj1: Conv2d,
    ln1: LayerNorm,
    proj2: Conv2d,
    ln2: LayerNorm,
}
impl Neck {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

pub struct PatchEmbed {
    proj: Conv2d,
    kernel_size: (usize, usize),
}

impl PatchEmbed {
    pub fn new(
        vb: VarBuilder,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        in_chans: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        };
        let proj = conv2d(in_chans, embed_dim, kernel_size, cfg, vb.pp("proj"))?;
        Ok(Self {
            proj,
            kernel_size: (kernel_size, kernel_size),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let (patch_h, patch_w) = self.kernel_size;
        if (h % patch_h) != 0 {
            candle::bail!("image height {h} is not a multiple of patch height {patch_h}")
        }
        if (w % patch_w) != 0 {
            candle::bail!("image width {w} is not a multiple of patch width {patch_w}")
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        // B C H W -> B H W C
        xs.reshape((b, c, h, w))?.permute((0, 2, 3, 1))
    }
}
