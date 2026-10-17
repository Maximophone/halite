//! Just enough JSON writing to match Python's `json.dumps` output byte for
//! byte (ensure_ascii escaping, chosen separators, dict key order).

pub enum Value {
    Null,
    Int(i64),
    /// An integer kept as its decimal digits (seeds can be any size).
    BigInt(String),
    Str(String),
    List(Vec<Value>),
    Object(Vec<(&'static str, Value)>),
}

/// `item` and `key` separators, e.g. (",", ":") or Python's default (", ", ": ").
pub struct Style {
    pub item: &'static str,
    pub key: &'static str,
}

pub const COMPACT: Style = Style { item: ",", key: ":" };
pub const DEFAULT: Style = Style { item: ", ", key: ": " };

pub fn dumps(v: &Value, style: &Style) -> String {
    let mut out = String::new();
    write(v, style, &mut out);
    out
}

fn write(v: &Value, st: &Style, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Int(i) => crate::game::push_int(out, *i),
        Value::BigInt(s) => out.push_str(s),
        Value::Str(s) => write_str(s, out),
        Value::List(items) => {
            out.push('[');
            for (k, i) in items.iter().enumerate() {
                if k > 0 {
                    out.push_str(st.item);
                }
                write(i, st, out);
            }
            out.push(']');
        }
        Value::Object(fields) => {
            out.push('{');
            for (k, (name, val)) in fields.iter().enumerate() {
                if k > 0 {
                    out.push_str(st.item);
                }
                write_str(name, out);
                out.push_str(st.key);
                write(val, st, out);
            }
            out.push('}');
        }
    }
}

/// Python's ensure_ascii string encoding.
pub fn write_str(s: &str, out: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    fn u16_escape(out: &mut String, c: u32) {
        out.push_str("\\u");
        for shift in [12, 8, 4, 0] {
            out.push(HEX[((c >> shift) & 0xf) as usize] as char);
        }
    }
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let c = ch as u32;
                if c >= 0x10000 {
                    let v = c - 0x10000;
                    u16_escape(out, 0xd800 | (v >> 10));
                    u16_escape(out, 0xdc00 | (v & 0x3ff));
                } else {
                    u16_escape(out, c);
                }
            }
        }
    }
    out.push('"');
}
