//! Reading the elements of a constant tensor, for display in the details pane.
//!
//! Tensors are read a window at a time rather than whole: a set of weights runs
//! to hundreds of millions of elements, and only the handful of rows on screen
//! are ever needed. Data kept in a separate file is opened on demand, so a
//! model can be explored without its weights alongside it.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use rten_onnx::onnx::DataType;

use crate::model::{Tensor, TensorData};

/// Why a tensor's elements could not be read.
#[derive(Debug)]
pub enum ReadError {
    /// The tensor carries no data at all.
    NoData,
    /// The tensor's data lives in a file that is not there. Models are often
    /// distributed with the weights in a separate download.
    MissingFile(PathBuf),
    Io(PathBuf, std::io::Error),
    /// A type this viewer cannot decode, such as the sub-byte formats.
    UnsupportedType(DataType),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::NoData => write!(f, "This tensor has no data stored with it."),
            ReadError::MissingFile(path) => write!(
                f,
                "The weights are in a separate file, which is not here:\n{}",
                path.display()
            ),
            ReadError::Io(path, err) => write!(f, "Could not read {}: {err}", path.display()),
            ReadError::UnsupportedType(dtype) => {
                write!(f, "Values of type {dtype} cannot be shown.")
            }
        }
    }
}

/// Where an external tensor's bytes are, resolved against the model's own
/// directory as ONNX specifies.
pub struct External {
    pub path: PathBuf,
    pub offset: u64,
    pub length: Option<u64>,
}

/// Read the `external_data` entries describing where a tensor is stored.
pub fn external_ref(tensor: &Tensor, base_dir: &Path) -> Option<External> {
    let TensorData::External { entries } = &tensor.data else {
        return None;
    };
    let entry = |key: &str| {
        entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    };

    Some(External {
        path: base_dir.join(entry("location").unwrap_or_default()),
        offset: entry("offset").and_then(|v| v.parse().ok()).unwrap_or(0),
        length: entry("length").and_then(|v| v.parse().ok()),
    })
}

/// Whether the tensor's values can be shown at all.
///
/// Checked before offering to display them, so that a tensor whose type is not
/// understood says so rather than opening onto an empty list.
pub fn readable(tensor: &Tensor) -> Result<(), ReadError> {
    if matches!(tensor.data, TensorData::Missing) {
        return Err(ReadError::NoData);
    }
    match &tensor.data {
        TensorData::Floats(_)
        | TensorData::Int32s(_)
        | TensorData::Int64s(_)
        | TensorData::Doubles(_) => Ok(()),
        _ => match element_size(tensor.dtype) {
            Some(_) => Ok(()),
            None => Err(ReadError::UnsupportedType(tensor.dtype)),
        },
    }
}

/// Format the `count` elements of `tensor` starting at `start`.
///
/// Returns fewer than `count` values at the end of the data. `base_dir` is the
/// directory of the model file, which external references are relative to.
pub fn read_elements(
    tensor: &Tensor,
    base_dir: &Path,
    start: u64,
    count: usize,
) -> Result<Vec<String>, ReadError> {
    let dtype = tensor.dtype;
    let start = start as usize;

    match &tensor.data {
        // The typed repeated fields are already decoded.
        TensorData::Floats(values) => Ok(slice(values, start, count)),
        TensorData::Int32s(values) => Ok(slice(values, start, count)),
        TensorData::Int64s(values) => Ok(slice(values, start, count)),
        TensorData::Doubles(values) => Ok(slice(values, start, count)),

        TensorData::Raw(bytes) => {
            let size = element_size(dtype).ok_or(ReadError::UnsupportedType(dtype))?;
            let from = (start * size).min(bytes.len());
            let to = (from + count * size).min(bytes.len());
            Ok(decode(&bytes[from..to], dtype, size))
        }

        TensorData::External { .. } => {
            let size = element_size(dtype).ok_or(ReadError::UnsupportedType(dtype))?;
            let external = external_ref(tensor, base_dir).ok_or(ReadError::NoData)?;
            let bytes = read_chunk(&external, (start * size) as u64, count * size)?;
            Ok(decode(&bytes, dtype, size))
        }

        TensorData::Missing => Err(ReadError::NoData),
    }
}

/// Read `len` bytes from `offset` within the external tensor's own span.
///
/// The file is opened per call rather than held open: reads happen only while a
/// tensor's values are on screen, and one open per frame costs far less than
/// keeping a handle for every weight in a model.
fn read_chunk(external: &External, offset: u64, len: usize) -> Result<Vec<u8>, ReadError> {
    let mut file = File::open(&external.path).map_err(|err| match err.kind() {
        ErrorKind::NotFound => ReadError::MissingFile(external.path.clone()),
        _ => ReadError::Io(external.path.clone(), err),
    })?;

    // Stay inside the tensor's own region of the file, which may hold every
    // weight in the model one after another.
    let len = match external.length {
        Some(length) => len.min(length.saturating_sub(offset) as usize),
        None => len,
    };

    let io = |err| ReadError::Io(external.path.clone(), err);
    file.seek(SeekFrom::Start(external.offset + offset))
        .map_err(io)?;

    let mut bytes = vec![0u8; len];
    let mut read = 0;
    while read < len {
        match file.read(&mut bytes[read..]).map_err(io)? {
            0 => break,
            n => read += n,
        }
    }
    bytes.truncate(read);
    Ok(bytes)
}

fn slice<T: std::fmt::Display>(values: &[T], start: usize, count: usize) -> Vec<String> {
    values
        .iter()
        .skip(start)
        .take(count)
        .map(|value| value.to_string())
        .collect()
}

fn decode(bytes: &[u8], dtype: DataType, size: usize) -> Vec<String> {
    bytes
        .chunks_exact(size)
        .map(|c| element(c, dtype))
        .collect()
}

/// Bytes per element, or `None` for types this viewer cannot decode: the
/// sub-byte formats, which do not sit on byte boundaries, and strings, which
/// are not stored as packed bytes at all.
fn element_size(dtype: DataType) -> Option<usize> {
    Some(match dtype {
        DataType::BOOL | DataType::INT8 | DataType::UINT8 => 1,
        DataType::FLOAT8E4M3FN
        | DataType::FLOAT8E4M3FNUZ
        | DataType::FLOAT8E5M2
        | DataType::FLOAT8E5M2FNUZ
        | DataType::FLOAT8E8M0 => 1,
        DataType::BFLOAT16 | DataType::FLOAT16 | DataType::INT16 | DataType::UINT16 => 2,
        DataType::FLOAT | DataType::INT32 | DataType::UINT32 => 4,
        DataType::DOUBLE | DataType::INT64 | DataType::UINT64 | DataType::COMPLEX64 => 8,
        DataType::COMPLEX128 => 16,
        _ => return None,
    })
}

/// Format one element from its packed little-endian bytes.
fn element(bytes: &[u8], dtype: DataType) -> String {
    fn array<const N: usize>(bytes: &[u8]) -> [u8; N] {
        let mut out = [0u8; N];
        out.copy_from_slice(&bytes[..N]);
        out
    }

    match dtype {
        DataType::BOOL => (bytes[0] != 0).to_string(),
        DataType::INT8 => (bytes[0] as i8).to_string(),
        DataType::UINT8 => bytes[0].to_string(),
        DataType::INT16 => i16::from_le_bytes(array(bytes)).to_string(),
        DataType::UINT16 => u16::from_le_bytes(array(bytes)).to_string(),
        DataType::INT32 => i32::from_le_bytes(array(bytes)).to_string(),
        DataType::UINT32 => u32::from_le_bytes(array(bytes)).to_string(),
        DataType::INT64 => i64::from_le_bytes(array(bytes)).to_string(),
        DataType::UINT64 => u64::from_le_bytes(array(bytes)).to_string(),
        DataType::FLOAT => f32::from_le_bytes(array(bytes)).to_string(),
        DataType::DOUBLE => f64::from_le_bytes(array(bytes)).to_string(),
        DataType::FLOAT16 => f16_to_f32(u16::from_le_bytes(array(bytes))).to_string(),
        // A bfloat16 is the top half of the f32 with the same value.
        DataType::BFLOAT16 => {
            f32::from_bits((u16::from_le_bytes(array(bytes)) as u32) << 16).to_string()
        }
        DataType::FLOAT8E4M3FN | DataType::FLOAT8E4M3FNUZ => float8(bytes[0], 4, dtype).to_string(),
        DataType::FLOAT8E5M2 | DataType::FLOAT8E5M2FNUZ => float8(bytes[0], 5, dtype).to_string(),
        DataType::FLOAT8E8M0 => {
            // An exponent on its own, with no sign or mantissa.
            let exponent = bytes[0] as i32 - 127;
            (2.0f32).powi(exponent).to_string()
        }
        DataType::COMPLEX64 => {
            let (re, im) = (array::<4>(&bytes[..4]), array::<4>(&bytes[4..]));
            format!("{}+{}i", f32::from_le_bytes(re), f32::from_le_bytes(im))
        }
        DataType::COMPLEX128 => {
            let (re, im) = (array::<8>(&bytes[..8]), array::<8>(&bytes[8..]));
            format!("{}+{}i", f64::from_le_bytes(re), f64::from_le_bytes(im))
        }
        _ => String::from("?"),
    }
}

/// Widen an IEEE half to a single.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits as u32 & 0x3ff;

    match exponent {
        // Zero and the subnormals, which have no implicit leading one. Scaling
        // the mantissa as a fraction of 2^-14 gets the value without having to
        // renormalize it by hand.
        0 => f32::from_bits(sign) + (mantissa as f32) * 2.0f32.powi(-24) * signum(sign),
        // Infinity and the NaNs.
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent as u32 + 112) << 23) | (mantissa << 13)),
    }
}

fn signum(sign: u32) -> f32 {
    if sign == 0 { 1.0 } else { -1.0 }
}

/// Widen one of the 8-bit float formats, given the width of its exponent.
///
/// The `UZ` variants have no signed zero and shift the exponent by one, which
/// is the only difference that matters here.
fn float8(byte: u8, exponent_bits: u32, dtype: DataType) -> f32 {
    let mantissa_bits = 7 - exponent_bits;
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((byte >> mantissa_bits) & ((1 << exponent_bits) - 1) as u8) as i32;
    let mantissa = (byte & ((1 << mantissa_bits) - 1) as u8) as f32;

    let unsigned_zero = matches!(dtype, DataType::FLOAT8E4M3FNUZ | DataType::FLOAT8E5M2FNUZ);
    let bias: i32 = if unsigned_zero {
        1 << (exponent_bits - 1)
    } else {
        (1 << (exponent_bits - 1)) - 1
    };
    let scale = 2.0f32.powi(mantissa_bits as i32);

    if exponent == 0 {
        // Subnormal: no implicit leading one.
        sign * (mantissa / scale) * 2.0f32.powi(1 - bias)
    } else {
        sign * (1.0 + mantissa / scale) * 2.0f32.powi(exponent - bias)
    }
}

/// Describe where an element sits, as `[i, j, k]`.
pub fn index_label(dims: &[i64], mut flat: u64) -> String {
    if dims.len() <= 1 {
        return format!("[{flat}]");
    }

    let mut indices = vec![0u64; dims.len()];
    for (axis, dim) in dims.iter().enumerate().rev() {
        let extent = (*dim).max(1) as u64;
        indices[axis] = flat % extent;
        flat /= extent;
    }

    let mut label = String::from("[");
    for (axis, index) in indices.iter().enumerate() {
        if axis > 0 {
            label.push_str(", ");
        }
        let _ = write!(label, "{index}");
    }
    label.push(']');
    label
}

#[cfg(test)]
mod tests {
    use super::{element, f16_to_f32, index_label, read_elements};
    use crate::model::{Tensor, TensorData};
    use rten_onnx::onnx::DataType;
    use std::path::Path;

    fn tensor(dtype: DataType, dims: &[i64], data: TensorData) -> Tensor {
        Tensor {
            dtype,
            dims: dims.to_vec(),
            data,
        }
    }

    #[test]
    fn test_reads_a_window_of_raw_data() {
        let bytes: Vec<u8> = (0..8i32).flat_map(|v| (v * 2).to_le_bytes()).collect();
        let tensor = tensor(DataType::INT32, &[8], TensorData::Raw(bytes));

        let read = |start, count| read_elements(&tensor, Path::new(""), start, count).unwrap();
        assert_eq!(read(0, 3), ["0", "2", "4"]);
        assert_eq!(read(5, 3), ["10", "12", "14"]);
        // A window running past the end returns what there is.
        assert_eq!(read(6, 8), ["12", "14"]);
        assert!(read(8, 4).is_empty());
    }

    #[test]
    fn test_reads_typed_fields() {
        let tensor = tensor(
            DataType::FLOAT,
            &[3],
            TensorData::Floats(vec![1.5, 2.5, 3.5]),
        );
        let read = read_elements(&tensor, Path::new(""), 1, 2).unwrap();
        assert_eq!(read, ["2.5", "3.5"]);
    }

    #[test]
    fn test_missing_external_file_is_reported() {
        let tensor = tensor(
            DataType::FLOAT,
            &[4],
            TensorData::External {
                entries: vec![("location".to_string(), "weights.bin".to_string())],
            },
        );
        let err = read_elements(&tensor, Path::new("/nowhere"), 0, 4).unwrap_err();
        assert!(
            err.to_string().contains("/nowhere/weights.bin"),
            "should name the file it wanted: {err}"
        );
    }

    #[test]
    fn test_half_precision() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555).to_string(), "0.33325195");
        // The smallest subnormal, and infinity.
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0xfc00).is_infinite() && f16_to_f32(0xfc00) < 0.0);
    }

    #[test]
    fn test_element_formats() {
        assert_eq!(element(&[1], DataType::BOOL), "true");
        assert_eq!(element(&[0xff], DataType::INT8), "-1");
        assert_eq!(element(&[0xff], DataType::UINT8), "255");
        assert_eq!(element(&0.5f32.to_le_bytes(), DataType::FLOAT), "0.5");
        // A bfloat16 holding 1.0 is the top half of 1.0f32.
        assert_eq!(element(&[0x80, 0x3f], DataType::BFLOAT16), "1");
    }

    #[test]
    fn test_index_labels() {
        assert_eq!(index_label(&[8], 5), "[5]");
        assert_eq!(index_label(&[2, 3], 4), "[1, 1]");
        assert_eq!(index_label(&[2, 3, 4], 13), "[1, 0, 1]");
    }
}
