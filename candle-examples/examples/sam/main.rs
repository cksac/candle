mod image_encoder;
mod mask_decoder;
mod model;
mod prompt_encoder;

use candle::{shape::D, Device, Result, Tensor};

fn main() -> Result<()> {
    // 3xH*W
    let mut x = Tensor::rand(0., 255., (3, 2, 2), &Device::Cpu).unwrap();
    dbg!(&x.shape());

    x = x.pad_with_zeros(D::Minus1, 0, 2).unwrap();
    dbg!(&x.shape());

    x = x.pad_with_zeros(D::Minus2, 0, 2).unwrap();
    dbg!(&x.shape());

    println!("{}", x);

    println!("{:?}", x.shape());

    x = x.gt(&(&x.ones_like()? * 125f64)?)?;

    println!("{}", x);

    x = x.narrow(D::Minus1, 0, 2)?;
    x = x.narrow(D::Minus2, 0, 2)?;

    println!("{}", x);

    Ok(())
}
