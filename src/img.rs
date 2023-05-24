use zune_jpeg::errors::DecodeErrors;
use zune_jpeg::zune_core::result::DecodingResult;
use zune_jpeg::{JpegDecoder, ImageInfo};

use zune_png::error::PngDecodeErrors;
use zune_png::{PngDecoder, PngInfo};

use crate::parsing::array_types::{Base, Array};




fn shape_from_info(info: ImageInfo) -> Vec<usize> {
    vec![info.height as usize, info.width as usize, 3]
}


pub fn decode_jpeg_bytes(jpeg_bytes: &[u8]) -> Result<Array<u8>, DecodeErrors> {
    let mut decoder = JpegDecoder::new(jpeg_bytes);
    let result = decoder.decode();
    let info = decoder.info();
    match result {
        Ok(pixels) => {
            let shape = shape_from_info(info.unwrap());
            Ok(Array(Base::Array(pixels), Some(shape)))
        },
        Err(err) => Err(err)
    }
}


fn shape_from_png_info(info: &PngInfo) -> Vec<usize> {
    vec![info.height as usize, info.width as usize, 4]
}


pub fn decode_png_bytes(png_bytes: &[u8]) -> Result<Array<u8>, PngDecodeErrors> {
    let mut decoder = PngDecoder::new(png_bytes);
    let result = decoder.decode();
    let maybe_info = decoder.get_info();
    match result {
        Ok(decoding) => {
            let shape = shape_from_png_info(maybe_info.unwrap());
            match decoding {
                DecodingResult::U8(pixels) => Ok(Array(Base::Array(pixels), Some(shape))),
                _ => panic!("other variants not supported yet")
            }
        },
        Err(err) => Err(err)
    }
}
