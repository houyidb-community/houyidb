//! 静态加密（P4.7.3）：AES-256-GCM 与可插拔密钥提供者。
//! Encryption at rest (P4.7.3): AES-256-GCM plus a pluggable key provider.
//!
//! 威胁模型 / Threat model:
//! 防磁盘/备份被窃取后的离线读取；不防运行中主机被攻破（密钥在内存），
//! 也不隐藏 LSN、长度、文件名等元数据。
//! EN: protects data at rest (stolen disk/backup), not a compromised running
//! host; metadata (LSNs, lengths, filenames) stays visible.
//!
//! 算法 / Algorithm: AES-256-GCM，每条记录/快照独立 12B 随机 nonce，16B tag。
//! nonce 绝不复用；随机源为 `rand::rngs::OsRng`。
//! EN: AES-256-GCM with a fresh random 12-byte nonce per record/snapshot.
//!
//! 密钥管理 / Key management:
//! - [`KeyProvider`] 是 KMS/HSM 的接入点；v1 提供本地 keyring 文件实现。
//! - 文件格式：每行 `id=64位hex`，可选 `active=<id>`；也接受单行裸 hex
//!   （id 默认 `local-1`）。密钥 id ≤16 个 ASCII 可打印字符（写进文件头）。
//! - 密钥用 `Zeroizing` 包裹，离开作用域自动清零；Debug 输出打码。
//!
//! EN: keyring file lines `id=hex`, optional `active=id`; ids are at most 16
//! printable ASCII chars. Keys are zeroized on drop and redacted in Debug.
//!
//! 限制 / Limits: v1 不做在线批量重加密、外部 KMS SDK、国密/FIPS。
//! 部署与密钥备份见 docs/DEPLOYMENT_GUIDE.md。
//! EN: no online bulk re-encryption, external KMS SDKs, or SM4/FIPS in v1.
//! See docs/DEPLOYMENT_GUIDE.en.md for deployment and key backups.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::RngCore;
use zeroize::Zeroizing;

/// 密钥字节数（AES-256）。 / AES-256 key length.
/// EN: Key size in bytes (AES-256).
pub const KEY_LEN: usize = 32;
/// GCM nonce 字节数。 / GCM nonce length.
/// EN: GCM nonce length in bytes.
pub const NONCE_LEN: usize = 12;
/// GCM tag 字节数。 / GCM tag length.
/// EN: GCM tag length in bytes.
pub const TAG_LEN: usize = 16;
/// 文件头中 key id 的定长槽位。 / Fixed key-id slot in file headers.
/// EN: Fixed-size key-id slot in file headers.
pub const KEY_ID_LEN: usize = 16;

/// 一份密钥材料。 / One key material.
/// EN: One key material.
#[derive(Clone)]
pub struct KeyMaterial {
    pub id: String,
    pub key: Zeroizing<[u8; KEY_LEN]>,
}

impl std::fmt::Debug for KeyMaterial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 绝不让密钥进日志。 / never let the key reach logs.
        // EN: Never let key material reach logs.
        f.debug_struct("KeyMaterial")
            .field("id", &self.id)
            .field("key", &"<redacted>")
            .finish()
    }
}

/// 密钥提供者（本地文件 / 将来的 KMS/HSM 实现）。
/// Key provider (local file today; KMS/HSM later).
pub trait KeyProvider: Send + Sync {
    /// 新写入使用的活动密钥。 / Active key for new writes.
    /// EN: Active key used for new writes.
    fn active(&self) -> io::Result<KeyMaterial>;
    /// 按 id 取旧密钥（读旧段/旧快照）。 / Look up old keys by id.
    /// EN: Looks up an old key by id (for old segments/snapshots).
    fn get(&self, id: &str) -> io::Result<Option<KeyMaterial>>;
    /// 全部密钥 id（构建 cipher 缓存用）。 / All key ids (for the cipher cache).
    /// EN: All key ids (used to build the cipher cache).
    fn ids(&self) -> io::Result<Vec<String>>;
}

/// 本地 keyring 文件实现。 / Local keyring backed by a file.
/// EN: Local keyring file implementation.
pub struct LocalKeyProvider {
    active: KeyMaterial,
    keys: HashMap<String, KeyMaterial>,
}

impl std::fmt::Debug for LocalKeyProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalKeyProvider")
            .field("active", &self.active.id)
            .field("keys", &self.keys.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl LocalKeyProvider {
    /// 从 keyring 文件加载；`active_override` 优先于文件内的 `active=`。
    /// EN: load a keyring file; `active_override` wins over the file's `active=`.
    pub fn from_file(path: &Path, active_override: Option<&str>) -> io::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("read encryption keyring {}: {e}", path.display()),
            )
        })?;
        Self::parse(&text, active_override)
    }

    /// 解析 keyring 文本（纯函数，便于测试）。 / Parse keyring text (testable).
    /// EN: Parses keyring text (pure function, easy to unit-test).
    pub fn parse(text: &str, active_override: Option<&str>) -> io::Result<Self> {
        let mut keys: HashMap<String, KeyMaterial> = HashMap::new();
        let mut file_active: Option<String> = None;
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (id, hex) = match line.split_once('=') {
                Some(("active", value)) => {
                    file_active = Some(value.trim().to_string());
                    continue;
                }
                Some((id, value)) => (id.trim().to_string(), value.trim()),
                None => ("local-1".to_string(), line),
            };
            validate_key_id(&id).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("keyring line {}: {e}", lineno + 1),
                )
            })?;
            let bytes = decode_hex_key(hex).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("keyring line {} (id {id}): {e}", lineno + 1),
                )
            })?;
            if keys.contains_key(&id) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("keyring: duplicate key id {id:?}"),
                ));
            }
            keys.insert(
                id.clone(),
                KeyMaterial {
                    id,
                    key: Zeroizing::new(bytes),
                },
            );
        }
        if keys.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "keyring: no keys found",
            ));
        }
        let active_id = active_override
            .map(str::to_string)
            .or(file_active)
            .unwrap_or_else(|| keys.keys().next().cloned().expect("non-empty"));
        let active = keys.get(&active_id).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("keyring: active key id {active_id:?} not found"),
            )
        })?;
        Ok(Self { active, keys })
    }
}

impl KeyProvider for LocalKeyProvider {
    fn active(&self) -> io::Result<KeyMaterial> {
        Ok(self.active.clone())
    }

    fn get(&self, id: &str) -> io::Result<Option<KeyMaterial>> {
        Ok(self.keys.get(id).cloned())
    }

    fn ids(&self) -> io::Result<Vec<String>> {
        Ok(self.keys.keys().cloned().collect())
    }
}

fn validate_key_id(id: &str) -> Result<(), String> {
    if id.is_empty() {
        return Err("empty key id".to_string());
    }
    if id.len() > KEY_ID_LEN {
        return Err(format!("key id {id:?} longer than {KEY_ID_LEN} bytes"));
    }
    if !id.bytes().all(|b| b.is_ascii_graphic() && b != b'=') {
        return Err(format!("key id {id:?} must be printable ASCII without '='"));
    }
    Ok(())
}

fn decode_hex_key(hex: &str) -> Result<[u8; KEY_LEN], String> {
    let hex = hex.trim();
    if hex.len() != KEY_LEN * 2 {
        return Err(format!(
            "key must be {} hex chars, got {}",
            KEY_LEN * 2,
            hex.len()
        ));
    }
    let mut out = [0u8; KEY_LEN];
    for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
        let s = std::str::from_utf8(chunk).map_err(|_| "key is not ASCII hex")?;
        out[i] = u8::from_str_radix(s, 16).map_err(|_| "key is not valid hex")?;
    }
    Ok(out)
}

/// 加密句柄：活动密钥 + 全部历史密钥的 cipher 缓存。
/// Encryption handle: active key plus cached ciphers for every known key.
pub struct Encryption {
    active_id: String,
    ciphers: HashMap<String, Aes256Gcm>,
}

impl std::fmt::Debug for Encryption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encryption")
            .field("active_id", &self.active_id)
            .field("cipher_count", &self.ciphers.len())
            .finish()
    }
}

impl Encryption {
    /// 从 provider 构建；校验活动密钥存在。 / Build from a provider.
    /// EN: Builds from a provider; validates that the active key exists.
    pub fn from_provider(provider: Arc<dyn KeyProvider>) -> io::Result<Self> {
        let active = provider.active()?;
        let mut ciphers = HashMap::new();
        for id in provider.ids()? {
            if let Some(material) = provider.get(&id)? {
                let cipher = Aes256Gcm::new_from_slice(&material.key[..])
                    .map_err(|e| io::Error::other(format!("invalid AES-256 key: {e}")))?;
                ciphers.insert(id, cipher);
            }
        }
        if !ciphers.contains_key(&active.id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "active encryption key {:?} missing from provider",
                    active.id
                ),
            ));
        }
        Ok(Self {
            active_id: active.id,
            ciphers,
        })
    }

    /// 从本地 keyring 文件构建。 / Build from a local keyring file.
    /// EN: Builds from a local keyring file.
    pub fn from_key_file(path: &Path, active_override: Option<&str>) -> io::Result<Self> {
        let provider = LocalKeyProvider::from_file(path, active_override)?;
        Self::from_provider(Arc::new(provider))
    }

    pub fn active_id(&self) -> &str {
        &self.active_id
    }

    pub fn has_key(&self, id: &str) -> bool {
        self.ciphers.contains_key(id)
    }

    /// 加密明文，返回 `nonce‖ciphertext‖tag`。 / Encrypt; returns nonce||ct||tag.
    /// EN: Encrypts plaintext; returns nonce||ciphertext||tag.
    pub fn encrypt(&self, plaintext: &[u8]) -> io::Result<Vec<u8>> {
        let cipher = self.ciphers.get(&self.active_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "active key missing from cipher cache",
            )
        })?;
        let mut nonce = [0u8; NONCE_LEN];
        rand::rngs::OsRng.fill_bytes(&mut nonce);
        let ct = cipher
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .map_err(|e| io::Error::other(format!("encryption failed: {e}")))?;
        let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ct);
        Ok(out)
    }

    /// 按文件头里的 key id 解密 `nonce‖ciphertext‖tag`。
    /// EN: decrypt with the key id recorded in the file header.
    pub fn decrypt(&self, key_id: &str, blob: &[u8]) -> io::Result<Vec<u8>> {
        let cipher = self.ciphers.get(key_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("encryption key id {key_id:?} not found in keyring"),
            )
        })?;
        if blob.len() < NONCE_LEN + TAG_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "encrypted blob shorter than nonce + tag",
            ));
        }
        let (nonce, ct) = blob.split_at(NONCE_LEN);
        cipher.decrypt(Nonce::from_slice(nonce), ct).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "AES-GCM authentication failed (wrong key or tampered data)",
            )
        })
    }
}

/// 把 key id 编码进定长文件头槽位（右侧补 0）。 / Encode a key id into 16 bytes.
/// EN: Encodes a key id into the fixed 16-byte header slot (zero-padded on the right).
pub fn encode_key_id(id: &str) -> io::Result<[u8; KEY_ID_LEN]> {
    validate_key_id(id).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut out = [0u8; KEY_ID_LEN];
    out[..id.len()].copy_from_slice(id.as_bytes());
    Ok(out)
}

/// 从定长槽位解码 key id（去尾部 0）。 / Decode a fixed-slot key id.
/// EN: Decodes a key id from the fixed slot (strips trailing zeros).
pub fn decode_key_id(bytes: &[u8]) -> io::Result<String> {
    if bytes.len() != KEY_ID_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("key id slot must be {KEY_ID_LEN} bytes"),
        ));
    }
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(KEY_ID_LEN);
    let s = std::str::from_utf8(&bytes[..end])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "key id is not valid UTF-8"))?;
    validate_key_id(s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_key(byte: u8) -> String {
        format!("{:02x}", byte).repeat(KEY_LEN)
    }

    fn provider(text: &str) -> Arc<dyn KeyProvider> {
        Arc::new(LocalKeyProvider::parse(text, None).expect("keyring"))
    }

    #[test]
    fn keyring_parses_ids_and_active() {
        let text = format!("active=k2\nk1={}\nk2={}\n", hex_key(1), hex_key(2));
        let p = LocalKeyProvider::parse(&text, None).unwrap();
        assert_eq!(p.active().unwrap().id, "k2");
        assert!(p.get("k1").unwrap().is_some());
        assert!(p.get("missing").unwrap().is_none());
        // 覆盖优先。
        // EN: The explicit override wins.
        let p2 = LocalKeyProvider::parse(&text, Some("k1")).unwrap();
        assert_eq!(p2.active().unwrap().id, "k1");
        // 单行裸 hex。
        // EN: Single-line bare hex.
        let p3 = LocalKeyProvider::parse(&hex_key(9), None).unwrap();
        assert_eq!(p3.active().unwrap().id, "local-1");
    }

    #[test]
    fn keyring_rejects_bad_input() {
        assert!(LocalKeyProvider::parse("", None).is_err());
        assert!(LocalKeyProvider::parse("k1=zz", None).is_err());
        assert!(LocalKeyProvider::parse(&format!("k1={}", hex_key(1)), Some("nope")).is_err());
        assert!(
            LocalKeyProvider::parse(&format!("toolongkeyid123456789={}", hex_key(1)), None)
                .is_err()
        );
        let dup = format!("k1={}\nk1={}", hex_key(1), hex_key(2));
        assert!(LocalKeyProvider::parse(&dup, None).is_err());
    }

    #[test]
    fn roundtrip_and_tamper_detection() {
        let enc = Encryption::from_provider(provider(&format!("k1={}", hex_key(7)))).unwrap();
        let plaintext = b"vector payload \x00\x01\x02";
        let blob = enc.encrypt(plaintext).unwrap();
        assert_ne!(&blob[..], plaintext);
        assert_eq!(enc.decrypt("k1", &blob).unwrap(), plaintext);

        // 篡改任意字节 → 认证失败。
        // EN: Tampering with any byte → authentication failure.
        let mut tampered = blob.clone();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0x01;
        assert!(enc.decrypt("k1", &tampered).is_err());
        // 未知 key id → 明确错误。
        // EN: Unknown key id → explicit error.
        assert!(enc.decrypt("nope", &blob).is_err());
    }

    #[test]
    fn different_keys_do_not_decrypt() {
        let a = Encryption::from_provider(provider(&format!("k1={}", hex_key(1)))).unwrap();
        let b = Encryption::from_provider(provider(&format!("k1={}", hex_key(2)))).unwrap();
        let blob = a.encrypt(b"secret").unwrap();
        assert!(b.decrypt("k1", &blob).is_err());
    }

    #[test]
    fn key_id_slots_roundtrip() {
        let slot = encode_key_id("prod-2026").unwrap();
        assert_eq!(decode_key_id(&slot).unwrap(), "prod-2026");
        assert!(encode_key_id("this-id-is-way-too-long").is_err());
        assert!(decode_key_id(&[0u8; KEY_ID_LEN]).is_err());
    }
}
