use std::path::Path;
use memmap2::Mmap;
use half::f16;
use crate::meta::MetaValue;

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
fn read_kv_value(c: &mut Cursor, vtype: u32) -> Result<MetaValue, GgufError> {
    match vtype {
        0 => {  // uint8
            let v = c.take(1)?[0] as u32;
            Ok(MetaValue::U32(v))
        }
        1 => {  // int8
            let v = c.take(1)?[0] as i32;
            Ok(MetaValue::I32(v))
        }
        2 => {  // uint16
            let v = u16::from_le_bytes(c.take(2)?.try_into().unwrap()) as u32;
            Ok(MetaValue::U32(v))
        }
        3 => {  // int16
            let v = i16::from_le_bytes(c.take(2)?.try_into().unwrap()) as i32;
            Ok(MetaValue::I32(v))
        }
        4 => {  // uint32
            let v = u32::from_le_bytes(c.take(4)?.try_into().unwrap());
            Ok(MetaValue::U32(v))
        }
        5 => {  // int32
            let v = i32::from_le_bytes(c.take(4)?.try_into().unwrap());
            Ok(MetaValue::I32(v))
        }
        6 => {  // float32
            let v = f32::from_le_bytes(c.take(4)?.try_into().unwrap());
            Ok(MetaValue::F32(v))
        }
        7 => {  // bool
            let v = c.take(1)?[0] != 0;
            Ok(MetaValue::Bool(v))
        }
        8 => {  // string
            let n = c.u64()? as usize;
            let s = c.take(n)?;
            Ok(MetaValue::Str(String::from_utf8_lossy(s).into_owned()))
        }
        10 => {  // uint64
            let v = u64::from_le_bytes(c.take(8)?.try_into().unwrap());
            Ok(MetaValue::U64(v))
        }
        11 => {  // int64
            let v = i64::from_le_bytes(c.take(8)?.try_into().unwrap());
            Ok(MetaValue::U64(v as u64))
        }
        12 => {  // float64
            let v = f64::from_le_bytes(c.take(8)?.try_into().unwrap());
            Ok(MetaValue::F32(v as f32))
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
                    Ok(MetaValue::ArrU32(arr))
                }
                5 => {  // array of int32
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let v = i32::from_le_bytes(c.take(4)?.try_into().unwrap());
                        arr.push(v);
                    }
                    Ok(MetaValue::ArrI32(arr))
                }
                6 => {  // array of float32
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let v = f32::from_le_bytes(c.take(4)?.try_into().unwrap());
                        arr.push(v);
                    }
                    Ok(MetaValue::ArrF32(arr))
                }
                8 => {  // array of string
                    let mut arr = Vec::with_capacity(n);
                    for _ in 0..n {
                        let s = c.gguf_string()?;
                        arr.push(s);
                    }
                    Ok(MetaValue::ArrStr(arr))
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
                let val = read_kv_value(&mut c, vtype)?;
                kv.push((key, val));
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
            other => return Err(GgufError::UnsupportedDtype(other)),
        };
        Ok(TensorF32 { name: name.to_string(), shape, data })
    }
}

// mmap ist über die Lebensdauer von GgufFile gültig; die Rohzeiger sind privat und
// werden nur über &self dereferenziert. Send/Sync sind nicht nötig für v1.
