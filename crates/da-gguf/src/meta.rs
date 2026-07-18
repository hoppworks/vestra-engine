#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    Bool(bool),
    Str(String),
    ArrU32(Vec<u32>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    ArrStr(Vec<String>),
}

impl super::GgufFile {
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.meta(key)? {
            MetaValue::U32(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn meta_i32(&self, key: &str) -> Option<i32> {
        match self.meta(key)? {
            MetaValue::I32(v) => Some(*v),
            MetaValue::U32(v) => Some(*v as i32),
            _ => None,
        }
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.meta(key)? {
            MetaValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        match self.meta(key)? {
            MetaValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_str(&self, key: &str) -> Option<String> {
        match self.meta(key)? {
            MetaValue::Str(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        match self.meta(key)? {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_arr_u32(&self, key: &str) -> Option<Vec<u32>> {
        match self.meta(key)? {
            MetaValue::ArrU32(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn meta_arr_i32(&self, key: &str) -> Option<Vec<i32>> {
        match self.meta(key)? {
            MetaValue::ArrI32(v) => Some(v.clone()),
            MetaValue::ArrU32(v) => Some(v.iter().map(|&x| x as i32).collect()),
            _ => None,
        }
    }

    pub fn meta_arr_f32(&self, key: &str) -> Option<Vec<f32>> {
        match self.meta(key)? {
            MetaValue::ArrF32(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn meta_arr_str(&self, key: &str) -> Option<Vec<String>> {
        match self.meta(key)? {
            MetaValue::ArrStr(v) => Some(v.clone()),
            _ => None,
        }
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }
}
