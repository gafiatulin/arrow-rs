// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::errors::AvroError;
use arrow_schema::ArrowError;
#[cfg(any(
    feature = "deflate",
    feature = "zstd",
    feature = "bzip2",
    feature = "xz"
))]
use std::io::{Read, Write};

/// The metadata key used for storing the JSON encoded [`CompressionCodec`]
pub const CODEC_METADATA_KEY: &str = "avro.codec";

/// Defines a valid range of compression levels for a codec.
trait CompressionLevel<T: std::fmt::Display + std::cmp::PartialOrd> {
    const MINIMUM_LEVEL: T;
    const MAXIMUM_LEVEL: T;

    fn is_valid_level(level: T) -> Result<(), ArrowError> {
        let range = Self::MINIMUM_LEVEL..=Self::MAXIMUM_LEVEL;
        if range.contains(&level) {
            Ok(())
        } else {
            Err(ArrowError::InvalidArgumentError(format!(
                "compression level {} out of range {}..={}",
                level,
                range.start(),
                range.end()
            )))
        }
    }
}

/// Compression level for [`CompressionCodec::Deflate`].
///
/// Range `0..=9`. Higher values produce smaller output at the cost of speed.
/// `0` disables compression. The default of `6` matches the `flate2` /
/// `miniz_oxide` backend default.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct DeflateLevel(u32);

impl CompressionLevel<u32> for DeflateLevel {
    const MINIMUM_LEVEL: u32 = 0;
    const MAXIMUM_LEVEL: u32 = 9;
}

impl Default for DeflateLevel {
    fn default() -> Self {
        Self(6)
    }
}

impl DeflateLevel {
    /// Try to construct a [`DeflateLevel`] from a raw `u32` in `0..=9`.
    pub fn try_new(level: u32) -> Result<Self, ArrowError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the raw level value.
    pub fn compression_level(&self) -> u32 {
        self.0
    }
}

/// Compression level for [`CompressionCodec::ZStandard`].
///
/// Range `1..=22`. The default of `3` matches the `zstd` library default
/// (the level previously implied by passing `0` to `zstd::Encoder::new`).
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct ZstdLevel(i32);

impl CompressionLevel<i32> for ZstdLevel {
    const MINIMUM_LEVEL: i32 = 1;
    const MAXIMUM_LEVEL: i32 = 22;
}

impl Default for ZstdLevel {
    fn default() -> Self {
        Self(3)
    }
}

impl ZstdLevel {
    /// Try to construct a [`ZstdLevel`] from a raw `i32` in `1..=22`.
    pub fn try_new(level: i32) -> Result<Self, ArrowError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the raw level value.
    pub fn compression_level(&self) -> i32 {
        self.0
    }
}

/// Compression level for [`CompressionCodec::Bzip2`].
///
/// Range `1..=9`. Each step represents a 100 KiB block-size increment.
/// The default of `9` matches Avro Java (which uses
/// `BZip2CompressorOutputStream`'s no-arg constructor = `MAX_BLOCKSIZE`)
/// and the `bzip2` CLI's default of `-9`.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Bzip2Level(u32);

impl CompressionLevel<u32> for Bzip2Level {
    const MINIMUM_LEVEL: u32 = 1;
    const MAXIMUM_LEVEL: u32 = 9;
}

impl Default for Bzip2Level {
    fn default() -> Self {
        Self(9)
    }
}

impl Bzip2Level {
    /// Try to construct a [`Bzip2Level`] from a raw `u32` in `1..=9`.
    pub fn try_new(level: u32) -> Result<Self, ArrowError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the raw level value.
    pub fn compression_level(&self) -> u32 {
        self.0
    }
}

/// Compression level for [`CompressionCodec::Xz`].
///
/// Range `0..=9`. The default of `6` matches the `xz` / `liblzma` default.
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct XzLevel(u32);

impl CompressionLevel<u32> for XzLevel {
    const MINIMUM_LEVEL: u32 = 0;
    const MAXIMUM_LEVEL: u32 = 9;
}

impl Default for XzLevel {
    fn default() -> Self {
        Self(6)
    }
}

impl XzLevel {
    /// Try to construct an [`XzLevel`] from a raw `u32` in `0..=9`.
    pub fn try_new(level: u32) -> Result<Self, ArrowError> {
        Self::is_valid_level(level).map(|_| Self(level))
    }

    /// Returns the raw level value.
    pub fn compression_level(&self) -> u32 {
        self.0
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// Supported compression codecs for Avro data
///
/// Avro supports multiple compression formats for data blocks.
/// This enum represents the compression codecs available in this implementation.
///
/// Codecs that take a compression level carry a typed level wrapper
/// (e.g. [`DeflateLevel`]). Use the `Default` impl on those wrappers to
/// get the codec's natural default level. The compression level is a
/// writer-side choice — it is not persisted in the OCF metadata, so the
/// reader does not need to know what level was used to compress a block.
pub enum CompressionCodec {
    /// Deflate compression (RFC 1951)
    Deflate(DeflateLevel),
    /// Snappy compression
    Snappy,
    /// ZStandard compression
    ZStandard(ZstdLevel),
    /// Bzip2 compression
    Bzip2(Bzip2Level),
    /// Xz compression
    Xz(XzLevel),
}

impl CompressionCodec {
    #[allow(unused_variables)]
    pub(crate) fn decompress(&self, block: &[u8]) -> Result<Vec<u8>, AvroError> {
        match self {
            #[cfg(feature = "deflate")]
            CompressionCodec::Deflate(_) => {
                let mut decoder = flate2::read::DeflateDecoder::new(block);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            #[cfg(not(feature = "deflate"))]
            CompressionCodec::Deflate(_) => Err(AvroError::ParseError(
                "Deflate codec requires deflate feature".to_string(),
            )),
            #[cfg(feature = "snappy")]
            CompressionCodec::Snappy => {
                // Each compressed block is followed by the 4-byte, big-endian CRC32
                // checksum of the uncompressed data in the block.
                let crc = &block[block.len() - 4..];
                let block = &block[..block.len() - 4];

                let mut decoder = snap::raw::Decoder::new();
                let decoded = decoder
                    .decompress_vec(block)
                    .map_err(|e| AvroError::External(Box::new(e)))?;

                let checksum = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC).checksum(&decoded);
                if checksum != u32::from_be_bytes(crc.try_into().unwrap()) {
                    return Err(AvroError::ParseError("Snappy CRC mismatch".to_string()));
                }
                Ok(decoded)
            }
            #[cfg(not(feature = "snappy"))]
            CompressionCodec::Snappy => Err(AvroError::ParseError(
                "Snappy codec requires snappy feature".to_string(),
            )),

            #[cfg(feature = "zstd")]
            CompressionCodec::ZStandard(_) => {
                let mut decoder = zstd::Decoder::new(block)?;
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| AvroError::External(Box::new(e)))?;
                Ok(out)
            }
            #[cfg(not(feature = "zstd"))]
            CompressionCodec::ZStandard(_) => Err(AvroError::ParseError(
                "ZStandard codec requires zstd feature".to_string(),
            )),
            #[cfg(feature = "bzip2")]
            CompressionCodec::Bzip2(_) => {
                let mut decoder = bzip2::read::BzDecoder::new(block);
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| AvroError::External(Box::new(e)))?;
                Ok(out)
            }
            #[cfg(not(feature = "bzip2"))]
            CompressionCodec::Bzip2(_) => Err(AvroError::ParseError(
                "Bzip2 codec requires bzip2 feature".to_string(),
            )),
            #[cfg(feature = "xz")]
            CompressionCodec::Xz(_) => {
                let mut decoder = xz::read::XzDecoder::new(block);
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| AvroError::External(Box::new(e)))?;
                Ok(out)
            }
            #[cfg(not(feature = "xz"))]
            CompressionCodec::Xz(_) => Err(AvroError::ParseError(
                "XZ codec requires xz feature".to_string(),
            )),
        }
    }

    #[allow(unused_variables)]
    pub(crate) fn compress(&self, data: &[u8]) -> Result<Vec<u8>, ArrowError> {
        match self {
            #[cfg(feature = "deflate")]
            CompressionCodec::Deflate(level) => {
                let mut encoder = flate2::write::DeflateEncoder::new(
                    Vec::new(),
                    flate2::Compression::new(level.compression_level()),
                );
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                Ok(compressed)
            }
            #[cfg(not(feature = "deflate"))]
            CompressionCodec::Deflate(_) => Err(ArrowError::ParseError(
                "Deflate codec requires deflate feature".to_string(),
            )),

            #[cfg(feature = "snappy")]
            CompressionCodec::Snappy => {
                let mut encoder = snap::raw::Encoder::new();
                // Allocate and compress in one step for efficiency
                let mut compressed = encoder
                    .compress_vec(data)
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                // Compute CRC32 (ISO‑HDLC poly) of **uncompressed** data
                let crc_val = crc::Crc::<u32>::new(&crc::CRC_32_ISO_HDLC).checksum(data);
                compressed.extend_from_slice(&crc_val.to_be_bytes());
                Ok(compressed)
            }
            #[cfg(not(feature = "snappy"))]
            CompressionCodec::Snappy => Err(ArrowError::ParseError(
                "Snappy codec requires snappy feature".to_string(),
            )),

            #[cfg(feature = "zstd")]
            CompressionCodec::ZStandard(level) => {
                let mut encoder = zstd::Encoder::new(Vec::new(), level.compression_level())
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                encoder.write_all(data)?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                Ok(compressed)
            }
            #[cfg(not(feature = "zstd"))]
            CompressionCodec::ZStandard(_) => Err(ArrowError::ParseError(
                "ZStandard codec requires zstd feature".to_string(),
            )),

            #[cfg(feature = "bzip2")]
            CompressionCodec::Bzip2(level) => {
                let mut encoder = bzip2::write::BzEncoder::new(
                    Vec::new(),
                    bzip2::Compression::new(level.compression_level()),
                );
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                Ok(compressed)
            }
            #[cfg(not(feature = "bzip2"))]
            CompressionCodec::Bzip2(_) => Err(ArrowError::ParseError(
                "Bzip2 codec requires bzip2 feature".to_string(),
            )),
            #[cfg(feature = "xz")]
            CompressionCodec::Xz(level) => {
                let mut encoder = xz::write::XzEncoder::new(Vec::new(), level.compression_level());
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                Ok(compressed)
            }
            #[cfg(not(feature = "xz"))]
            CompressionCodec::Xz(_) => Err(ArrowError::ParseError(
                "XZ codec requires xz feature".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflate_level_range() {
        assert_eq!(DeflateLevel::default().compression_level(), 6);
        assert_eq!(DeflateLevel::try_new(0).unwrap().compression_level(), 0);
        assert_eq!(DeflateLevel::try_new(9).unwrap().compression_level(), 9);
        assert!(DeflateLevel::try_new(10).is_err());
    }

    #[test]
    fn zstd_level_range() {
        assert_eq!(ZstdLevel::default().compression_level(), 3);
        assert!(ZstdLevel::try_new(0).is_err());
        assert_eq!(ZstdLevel::try_new(1).unwrap().compression_level(), 1);
        assert_eq!(ZstdLevel::try_new(22).unwrap().compression_level(), 22);
        assert!(ZstdLevel::try_new(23).is_err());
    }

    #[test]
    fn bzip2_level_range() {
        assert_eq!(Bzip2Level::default().compression_level(), 9);
        assert!(Bzip2Level::try_new(0).is_err());
        assert_eq!(Bzip2Level::try_new(1).unwrap().compression_level(), 1);
        assert_eq!(Bzip2Level::try_new(9).unwrap().compression_level(), 9);
        assert!(Bzip2Level::try_new(10).is_err());
    }

    #[test]
    fn xz_level_range() {
        assert_eq!(XzLevel::default().compression_level(), 6);
        assert_eq!(XzLevel::try_new(0).unwrap().compression_level(), 0);
        assert_eq!(XzLevel::try_new(9).unwrap().compression_level(), 9);
        assert!(XzLevel::try_new(10).is_err());
    }

    /// Higher deflate levels should produce strictly-no-larger output than
    /// lower levels on data with redundancy. Round-tripping at any level
    /// must yield identical bytes.
    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_levels_produce_smaller_output_at_higher_setting() {
        // Repeating payload so the level actually matters.
        let data: Vec<u8> = (0..4096).flat_map(|i: u32| i.to_le_bytes()).collect();

        let fast = CompressionCodec::Deflate(DeflateLevel::try_new(1).unwrap())
            .compress(&data)
            .unwrap();
        let best = CompressionCodec::Deflate(DeflateLevel::try_new(9).unwrap())
            .compress(&data)
            .unwrap();

        assert!(
            best.len() <= fast.len(),
            "level 9 ({} bytes) should not exceed level 1 ({} bytes)",
            best.len(),
            fast.len()
        );

        for raw in [&fast, &best] {
            let round_trip = CompressionCodec::Deflate(DeflateLevel::default())
                .decompress(raw)
                .unwrap();
            assert_eq!(round_trip, data);
        }
    }
}
