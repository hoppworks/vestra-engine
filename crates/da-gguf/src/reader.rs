use std::path::Path;
use memmap2::Mmap;
use half::f16;
use crate::meta::MetaValue;
use crate::q8_0::{self, TensorQ8_0, QK8_0};

#[derive(thiserror::Error, Debug)]
pub enum GgufError {
    #[error("io: {0}")] Io(#[from] std::io::Error),
    #[error("bad magic")] BadMagic,
    #[error("unsupported gguf version {0}")] UnsupportedVersion(u32),
    #[error("tensor not found: {0}")] TensorNotFound(String),
    #[error("unsupported dtype {0}")] UnsupportedDtype(u32),
    #[error("malformed: {0}")] Malformed(String),
}

pub struct TensorF32 { pub name: String, pub shape: Vec<i64>, pub data: Vec<f32> }

pub struct TensorInfo { pub name: String, dims: Vec<u64>, dtype: u32, offset: u64 }

pub struct GgufFile {
    _mmap: Mmap,
    pub kv: Vec<(String, MetaValue)>,
    pub tensors: Vec<TensorInfo>,
    data_start: usize,
}

// GGML dtype-Codes (Teilmenge v1): F32=0, F16=1, Q8_0=8.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q8_0: u32 = 8;

struct Cursor<'a> { b: &'a [u8], p: usize }
impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32, GgufError> { let v = u32::from_le_bytes(self.take(4)?.try_into().unwrap()); Ok(v) }
    fn u64(&mut self) -> Result<u64, GgufError> { let v = u64::from_le_bytes(self.take(8)?.try_into().unwrap()); Ok(v) }
    fn i32(&mut self) -> Result<i32, GgufError> { Ok(self.u32()? as i32) }
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.p + n > self.b.len() { return Err(GgufError::Malformed("eof".into())); }
        let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s)
    }
    fn gguf_string(&mut self) -> Result<String, GgufError> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
}

// KV-Value-Typen laut GGUF-Spec. Lies und gib den Wert zurück.
// Returns Ok(Some(value)) for supported types, Ok(None) for unsupported but skipped types.
fn read_kv_value(c: &mut Cursor, vtype: u32) -> Result<Option<MetaValue>, GgufError> {
    match vtype {
        0 => {  // uint8
            let v = c.take(1)?[0] as u32;
            Ok(Some(MetaValue::U32(v)))
        }
        1 => {  // int8
            let v = c.take(1)?[0] as i32;
            Ok(Some(MetaValue::I32(v)))
        }
        2 => {  // uint16
            let v = u16::from_le_bytes(c.take(2)?.try_into().unwrap()) as u32;
            Ok(Some(MetaValue::U32(v)))
        }
        3 => {  // int16
            let v = i16::from_le_bytes(c.take(2)?.try_into().unwrap()) as i32;
            Ok(Some(MetaValue::I32(v)))
        }
        4 => {  // uint32
            let v = u32::from_le_bytes(c.take(4)?.try_into().unwrap());
            Ok(Some(MetaValue::U32(v)))
        }
        5 => {  // int32
            let v = i32::from_le_bytes(c.take(4)?.try_into().unwrap());
            Ok(Some(MetaValue::I32(v)))
        }
        6 => {  // float32
            let v = f32::from_le_bytes(c.take(4)?.try_into().unwrap());
            Ok(Some(MetaValue::F32(v)))
        }
        7 => {  // bool
            let v = c.take(1)?[0] != 0;
            Ok(Some(MetaValue::Bool(v)))
        }
        8 => {  // string
            let n = c.u64()? as usize;
            let s = c.take(n)?;
            Ok(Some(MetaValue::Str(String::from_utf8_lossy(s).into_owned())))
        }
        10 => {  // uint64
            let v = u64::from_le_bytes(c.take(8)?.try_into().unwrap());
            Ok(Some(MetaValue::U64(v)))
        }
        11 => {  // int64
            let v = i64::from_le_bytes(c.take(8)?.try_into().unwrap());
            Ok(Some(MetaValue::U64(v as u64)))
        }
        12 => {  // float64
            let v = f64::from_le_bytes(c.take(8)?.try_into().unwrap());
            Ok(Some(MetaValue::F32(v as f32)))
        }
        9 => {  // array
            let elem = c.u32()?;
            let n = c.u64()? as usize;
            match elem {
                4 => {  // array of uint32
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let v = u32::from_le_bytes(c.take(4)?.try_into().unwrap());
                        arr.push(v);
                    }
                    Ok(Some(MetaValue::ArrU32(arr)))
                }
                5 => {  // array of int32
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let v = i32::from_le_bytes(c.take(4)?.try_into().unwrap());
                        arr.push(v);
                    }
                    Ok(Some(MetaValue::ArrI32(arr)))
                }
                6 => {  // array of float32
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let v = f32::from_le_bytes(c.take(4)?.try_into().unwrap());
                        arr.push(v);
                    }
                    Ok(Some(MetaValue::ArrF32(arr)))
                }
                8 => {  // array of string
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let s = c.gguf_string()?;
                        arr.push(s);
                    }
                    Ok(Some(MetaValue::ArrStr(arr)))
                }
                // Unsupported but spec-valid array element types: skip their bytes and return None
                0 | 1 | 2 | 3 | 7 | 10 | 11 | 12 => {
                    // Skip n elements of the given type
                    let bytes_per_elem = match elem {
                        0 | 1 | 7 => 1,      // uint8, int8, bool
                        2 | 3 => 2,           // uint16, int16
                        10 | 11 => 8,         // uint64, int64
                        12 => 8,              // float64
                        _ => unreachable!(),
                    };
                    c.take(n * bytes_per_elem)?;
                    Ok(None)
                }
                other => Err(GgufError::Malformed(format!("array element type {other}"))),
            }
        }
        other => Err(GgufError::Malformed(format!("kv type {other}"))),
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile, GgufError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cursor { b: &mmap[..], p: 0 };
        if c.take(4)? != b"GGUF" { return Err(GgufError::BadMagic); }
        let version = c.u32()?;
        if version != 2 && version != 3 { return Err(GgufError::UnsupportedVersion(version)); }
        let tensor_count = c.u64()?;
        let kv_count = c.u64()?;
        let mut alignment: u64 = 32;
        let mut kv = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = c.gguf_string()?;
            let vtype = c.u32()?;
            if key == "general.alignment" && vtype == 4 {
                alignment = c.u32()? as u64;
                // Store general.alignment as a normal KV entry
                kv.push((key, MetaValue::U32(alignment as u32)));
            } else {
                if let Some(val) = read_kv_value(&mut c, vtype)? {
                    kv.push((key, val));
                }
                // If read_kv_value returns None (unsupported but skipped type), don't insert entry
            }
        }
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = c.gguf_string()?;
            let n_dims = c.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims { dims.push(c.u64()?); }
            let dtype = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }
        // Datenblock beginnt am nächsten `alignment`-Vielfachen nach den Infos.
        let pad = (alignment - (c.p as u64 % alignment)) % alignment;
        let data_start = c.p + pad as usize;
        Ok(GgufFile { _mmap: mmap, kv, tensors, data_start })
    }

    fn info(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.tensors.iter().find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))
    }

    fn raw(&self) -> &[u8] { &self._mmap[..] }

    pub fn tensor_f32(&self, name: &str) -> Result<TensorF32, GgufError> {
        let ti = self.info(name)?;
        // Shape outer→inner wie parity.hpp: dims sind inner→outer gespeichert, also umdrehen.
        let shape: Vec<i64> = ti.dims.iter().rev().map(|&d| d as i64).collect();
        let n: usize = ti.dims.iter().map(|&d| d as usize).product();
        let base = self.data_start + ti.offset as usize;
        let bytes = self.raw();
        let data = match ti.dtype {
            GGML_F32 => {
                let end = base + n * 4;
                if end > bytes.len() {
                    return Err(GgufError::Malformed(format!(
                        "tensor '{}' data out of bounds: [{}, {}) exceeds file size {}",
                        name, base, end, bytes.len()
                    )));
                }
                bytes[base..end].chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            }
            GGML_F16 => {
                let end = base + n * 2;
                if end > bytes.len() {
                    return Err(GgufError::Malformed(format!(
                        "tensor '{}' data out of bounds: [{}, {}) exceeds file size {}",
                        name, base, end, bytes.len()
                    )));
                }
                bytes[base..end].chunks_exact(2)
                    .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect()
            }
            GGML_Q8_0 => {
                let nblocks = n / QK8_0;
                let end = base + nblocks * 34;
                if end > bytes.len() {
                    return Err(GgufError::Malformed(format!(
                        "tensor '{}' data out of bounds: [{}, {}) exceeds file size {}",
                        name, base, end, bytes.len()
                    )));
                }
                let blocks = q8_0::read_blocks(&bytes[base..end], n);
                let mut data = vec![0f32; n];
                q8_0::dequantize_q8_0(&blocks, &mut data);
                data
            }
            other => return Err(GgufError::UnsupportedDtype(other)),
        };
        Ok(TensorF32 { name: name.to_string(), shape, data })
    }

    pub fn tensor_q8_0(&self, name: &str) -> Result<TensorQ8_0, GgufError> {
        let ti = self.info(name)?;
        let shape: Vec<i64> = ti.dims.iter().rev().map(|&d| d as i64).collect();
        let n: usize = ti.dims.iter().map(|&d| d as usize).product();
        let base = self.data_start + ti.offset as usize;
        let bytes = self.raw();
        let nblocks = n / QK8_0;
        let end = base + nblocks * 34;
        if end > bytes.len() {
            return Err(GgufError::Malformed(format!(
                "tensor '{}' data out of bounds: [{}, {}) exceeds file size {}",
                name, base, end, bytes.len()
            )));
        }
        let blocks = q8_0::read_blocks(&bytes[base..end], n);
        Ok(TensorQ8_0 { shape, blocks })
    }
}

// mmap ist über die Lebensdauer von GgufFile gültig; die Rohzeiger sind privat und
// werden nur über &self dereferenziert. Send/Sync sind nicht nötig für v1.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_kv_value_unmodeled_array_skips_correctly() {
        // Regression test for bug where unmodeled array element types (e.g., bool, u16, u64, f64)
        // caused hard failures. After the fix, read_kv_value returns Ok(None) for unsupported but
        // valid element types, and the cursor advances past the array bytes correctly.
        //
        // This test constructs a KV byte sequence with:
        // 1. An array KV entry with unmodeled element type (bool=7) and a few elements
        // 2. A sentinel u32 KV entry with a known value immediately after
        //
        // If the cursor doesn't advance correctly after skipping the unmodeled array,
        // the sentinel parse will fail or produce wrong values.

        // Build test bytes manually: two KV entries

        // ===== First KV: array of bool (unmodeled, should be skipped) =====
        // key_len: 20 bytes for "unmodeled_bool_array"
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&20u64.to_le_bytes());
        bytes.extend_from_slice(b"unmodeled_bool_array");

        // vtype: 9 (array)
        bytes.extend_from_slice(&9u32.to_le_bytes());

        // elem_type: 7 (bool)
        bytes.extend_from_slice(&7u32.to_le_bytes());

        // elem_count: 3
        bytes.extend_from_slice(&3u64.to_le_bytes());

        // elements: 3 bool values (1 byte each): true, false, true
        bytes.push(0x01);
        bytes.push(0x00);
        bytes.push(0x01);

        // ===== Second KV: u32 sentinel =====
        // key_len: 8 bytes for "sentinel"
        bytes.extend_from_slice(&8u64.to_le_bytes());
        bytes.extend_from_slice(b"sentinel");

        // vtype: 4 (u32)
        bytes.extend_from_slice(&4u32.to_le_bytes());

        // value: 0x12345678
        bytes.extend_from_slice(&0x12345678u32.to_le_bytes());

        // ===== Parse both KV entries =====
        let mut cursor = Cursor { b: &bytes, p: 0 };

        // Parse first KV entry (unmodeled array)
        let key1 = cursor.gguf_string().expect("read key1");
        assert_eq!(key1, "unmodeled_bool_array");

        let vtype1 = cursor.u32().expect("read vtype1");
        assert_eq!(vtype1, 9);

        // Parse the unmodeled array; should return Ok(None) and advance cursor
        let result1 = read_kv_value(&mut cursor, vtype1);
        assert!(result1.is_ok(), "unmodeled array should not error, got: {:?}", result1);
        let opt_val1 = result1.expect("result is ok");
        assert!(opt_val1.is_none(), "unmodeled array should return None");

        // ===== Parse second KV entry (sentinel u32) =====
        let key2 = cursor.gguf_string().expect("read key2");
        assert_eq!(key2, "sentinel");

        let vtype2 = cursor.u32().expect("read vtype2");
        assert_eq!(vtype2, 4);

        // Parse the sentinel u32; must produce correct value
        let result2 = read_kv_value(&mut cursor, vtype2);
        assert!(result2.is_ok(), "sentinel u32 should parse, got: {:?}", result2);
        let opt_val2 = result2.expect("result is ok");
        assert!(opt_val2.is_some(), "sentinel u32 should produce Some value");

        if let Some(MetaValue::U32(val)) = opt_val2 {
            assert_eq!(val, 0x12345678, "sentinel u32 value mismatch");
        } else {
            panic!("expected MetaValue::U32, got {:?}", opt_val2);
        }

        // ===== Final sanity check =====
        // Cursor should have consumed exactly all bytes
        assert_eq!(cursor.p, bytes.len(), "cursor should be at end of bytes");
    }

    #[test]
    fn test_read_kv_value_array_of_u16_skips_correctly() {
        // Additional regression test: array of u16 (type 2) should be skipped.
        // u16 is 2 bytes per element, so 5 elements = 10 bytes to skip.

        let mut bytes = Vec::new();

        // ===== First KV: array of u16 (unmodeled) =====
        bytes.extend_from_slice(&19u64.to_le_bytes()); // key_len
        bytes.extend_from_slice(b"unmodeled_u16_array");

        bytes.extend_from_slice(&9u32.to_le_bytes()); // vtype: array
        bytes.extend_from_slice(&2u32.to_le_bytes()); // elem_type: u16
        bytes.extend_from_slice(&5u64.to_le_bytes()); // elem_count: 5

        // 5 u16 values: 0x1111, 0x2222, 0x3333, 0x4444, 0x5555
        bytes.extend_from_slice(&0x1111u16.to_le_bytes());
        bytes.extend_from_slice(&0x2222u16.to_le_bytes());
        bytes.extend_from_slice(&0x3333u16.to_le_bytes());
        bytes.extend_from_slice(&0x4444u16.to_le_bytes());
        bytes.extend_from_slice(&0x5555u16.to_le_bytes());

        // ===== Second KV: i32 sentinel =====
        bytes.extend_from_slice(&10u64.to_le_bytes()); // key_len
        bytes.extend_from_slice(b"sentinel_i");

        bytes.extend_from_slice(&5u32.to_le_bytes()); // vtype: i32
        bytes.extend_from_slice(&(-42i32).to_le_bytes()); // value: -42

        // Parse
        let mut cursor = Cursor { b: &bytes, p: 0 };

        let key1 = cursor.gguf_string().expect("read key1");
        assert_eq!(key1, "unmodeled_u16_array");

        let vtype1 = cursor.u32().expect("read vtype1");
        assert_eq!(vtype1, 9);

        let result1 = read_kv_value(&mut cursor, vtype1);
        assert!(result1.is_ok());
        assert!(result1.unwrap().is_none(), "array of u16 should return None");

        let key2 = cursor.gguf_string().expect("read key2");
        assert_eq!(key2, "sentinel_i");

        let vtype2 = cursor.u32().expect("read vtype2");
        assert_eq!(vtype2, 5);

        let result2 = read_kv_value(&mut cursor, vtype2);
        assert!(result2.is_ok());
        if let Some(MetaValue::I32(val)) = result2.unwrap() {
            assert_eq!(val, -42);
        } else {
            panic!("expected MetaValue::I32");
        }

        assert_eq!(cursor.p, bytes.len());
    }
}
