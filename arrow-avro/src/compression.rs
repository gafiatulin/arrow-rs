// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

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

/// Define a validated compression-level newtype.
///
/// Every level type is the same shape: a raw value that can only be built inside the range the
/// codec accepts, the range itself, and the codec's default.
macro_rules! compression_level {
    (
        $(#[$meta:meta])*
        $name:ident { $ty:ty, $min:expr, $max:expr, $default:expr }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
        pub struct $name($ty);

        impl $name {
            /// Lowest level the codec accepts.
            pub const MIN: $ty = $min;

            /// Highest level the codec accepts.
            pub const MAX: $ty = $max;

            /// Try to construct a level from a raw value.
            ///
            /// Fails if the value is outside [`MIN..=MAX`](Self::MIN).
            pub fn try_new(level: $ty) -> Result<Self, ArrowError> {
                let range = Self::MIN..=Self::MAX;
                if range.contains(&level) {
                    Ok(Self(level))
                } else {
                    Err(ArrowError::InvalidArgumentError(format!(
                        "compression level {} out of range {}..={}",
                        level,
                        range.start(),
                        range.end()
                    )))
                }
            }

            /// Returns the raw level value.
            pub fn compression_level(&self) -> $ty {
                self.0
            }
        }

        impl Default for $name {
            /// The codec's own default, which is always inside [`MIN..=MAX`](Self::MIN).
            fn default() -> Self {
                debug_assert!(
                    (Self::MIN..=Self::MAX).contains(&$default),
                    concat!(stringify!($name), " default is outside the codec's range")
                );
                Self($default)
            }
        }
    };
}

compression_level! {
    /// Compression level for [`CompressionCodec::Deflate`].
    ///
    /// Range `0..=9`. Higher values produce smaller output at the cost of speed.
    /// `0` disables compression. The default of `6` matches the `flate2` /
    /// `miniz_oxide` backend default.
    DeflateLevel { u32, 0, 9, 6 }
}

compression_level! {
    /// Compression level for [`CompressionCodec::ZStandard`].
    ///
    /// Negative levels select zstd's fast modes (down to `-131072`), `0` selects the library
    /// default, and positive levels up to `22` trade speed for size. The default of `3` is that
    /// library default made explicit, so it compresses exactly as passing `0` used to.
    ///
    /// The range is not advisory: `zstd::Encoder::new` rejects anything else, and
    /// `zstd_level_range_matches_the_crate` keeps the constants in step with the crate.
    ZstdLevel { i32, -131_072, 22, 3 }
}

compression_level! {
    /// Compression level for [`CompressionCodec::Bzip2`].
    ///
    /// Range `1..=9`; each step adds 100 KiB to the block size. The range is not advisory:
    /// `bzip2::Compression::new` panics outside it, which is why the only constructor is
    /// [`try_new`](Self::try_new). The default of `6` is `bzip2::Compression::default()`, i.e. the
    /// level the writer used before levels were configurable; Avro Java's
    /// `BZip2CompressorOutputStream` uses the largest block size instead, which is `9` here.
    Bzip2Level { u32, 1, 9, 6 }
}

compression_level! {
    /// Compression level for [`CompressionCodec::Xz`].
    ///
    /// Range `0..=9`. The default of `6` matches the `xz` / `liblzma` default.
    ///
    /// The range is not advisory: `xz::write::XzEncoder::new` unwraps the preset error, so an
    /// out-of-range level would panic inside the encoder.
    XzLevel { u32, 0, 9, 6 }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// Supported compression codecs for Avro data
///
/// Avro supports multiple compression formats for data blocks.
/// This enum represents the compression codecs available in this implementation.
///
/// The compression level is a writer-side choice — it is not persisted in the OCF metadata, so
/// the reader does not need to know what level was used to compress a block. Configure it on
/// [`WriterBuilder`](crate::writer::WriterBuilder), e.g.
/// [`with_deflate_level`](crate::writer::WriterBuilder::with_deflate_level).
pub enum CompressionCodec {
    /// Deflate compression (RFC 1951)
    Deflate,
    /// Snappy compression
    Snappy,
    /// ZStandard compression
    ZStandard,
    /// Bzip2 compression
    Bzip2,
    /// Xz compression
    Xz,
}

impl CompressionCodec {
    #[cfg_attr(
        not(any(
            feature = "deflate",
            feature = "snappy",
            feature = "zstd",
            feature = "bzip2",
            feature = "xz"
        )),
        expect(unused_variables)
    )]
    pub(crate) fn decompress(&self, block: &[u8]) -> Result<Vec<u8>, AvroError> {
        match self {
            #[cfg(feature = "deflate")]
            CompressionCodec::Deflate => {
                let mut decoder = flate2::read::DeflateDecoder::new(block);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
            #[cfg(not(feature = "deflate"))]
            CompressionCodec::Deflate => Err(AvroError::ParseError(
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
            CompressionCodec::ZStandard => {
                let mut decoder = zstd::Decoder::new(block)?;
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| AvroError::External(Box::new(e)))?;
                Ok(out)
            }
            #[cfg(not(feature = "zstd"))]
            CompressionCodec::ZStandard => Err(AvroError::ParseError(
                "ZStandard codec requires zstd feature".to_string(),
            )),
            #[cfg(feature = "bzip2")]
            CompressionCodec::Bzip2 => {
                let mut decoder = bzip2::read::BzDecoder::new(block);
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| AvroError::External(Box::new(e)))?;
                Ok(out)
            }
            #[cfg(not(feature = "bzip2"))]
            CompressionCodec::Bzip2 => Err(AvroError::ParseError(
                "Bzip2 codec requires bzip2 feature".to_string(),
            )),
            #[cfg(feature = "xz")]
            CompressionCodec::Xz => {
                let mut decoder = xz::read::XzDecoder::new(block);
                let mut out = Vec::new();
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| AvroError::External(Box::new(e)))?;
                Ok(out)
            }
            #[cfg(not(feature = "xz"))]
            CompressionCodec::Xz => Err(AvroError::ParseError(
                "XZ codec requires xz feature".to_string(),
            )),
        }
    }
}

/// A [`CompressionCodec`] paired with the level the writer compresses with.
///
/// The level is writer-side configuration rather than part of a codec's identity, so it is not
/// carried by [`CompressionCodec`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum LeveledCodec {
    Deflate(DeflateLevel),
    Snappy,
    ZStandard(ZstdLevel),
    Bzip2(Bzip2Level),
    Xz(XzLevel),
}

impl From<CompressionCodec> for LeveledCodec {
    fn from(codec: CompressionCodec) -> Self {
        match codec {
            CompressionCodec::Deflate => Self::Deflate(DeflateLevel::default()),
            CompressionCodec::Snappy => Self::Snappy,
            CompressionCodec::ZStandard => Self::ZStandard(ZstdLevel::default()),
            CompressionCodec::Bzip2 => Self::Bzip2(Bzip2Level::default()),
            CompressionCodec::Xz => Self::Xz(XzLevel::default()),
        }
    }
}

impl LeveledCodec {
    #[cfg_attr(
        not(any(
            feature = "deflate",
            feature = "snappy",
            feature = "zstd",
            feature = "bzip2",
            feature = "xz"
        )),
        expect(unused_variables)
    )]
    pub(crate) fn compress(&self, data: &[u8]) -> Result<Vec<u8>, ArrowError> {
        match self {
            #[cfg(feature = "deflate")]
            LeveledCodec::Deflate(level) => {
                let mut encoder = flate2::write::DeflateEncoder::new(
                    Vec::new(),
                    flate2::Compression::new(level.compression_level()),
                );
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                Ok(compressed)
            }
            #[cfg(not(feature = "deflate"))]
            LeveledCodec::Deflate(_) => Err(ArrowError::ParseError(
                "Deflate codec requires deflate feature".to_string(),
            )),

            #[cfg(feature = "snappy")]
            LeveledCodec::Snappy => {
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
            LeveledCodec::Snappy => Err(ArrowError::ParseError(
                "Snappy codec requires snappy feature".to_string(),
            )),

            #[cfg(feature = "zstd")]
            LeveledCodec::ZStandard(level) => {
                let mut encoder = zstd::Encoder::new(Vec::new(), level.compression_level())
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                encoder.write_all(data)?;
                let compressed = encoder
                    .finish()
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                Ok(compressed)
            }
            #[cfg(not(feature = "zstd"))]
            LeveledCodec::ZStandard(_) => Err(ArrowError::ParseError(
                "ZStandard codec requires zstd feature".to_string(),
            )),

            #[cfg(feature = "bzip2")]
            LeveledCodec::Bzip2(level) => {
                let mut encoder = bzip2::write::BzEncoder::new(
                    Vec::new(),
                    bzip2::Compression::new(level.compression_level()),
                );
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                Ok(compressed)
            }
            #[cfg(not(feature = "bzip2"))]
            LeveledCodec::Bzip2(_) => Err(ArrowError::ParseError(
                "Bzip2 codec requires bzip2 feature".to_string(),
            )),
            #[cfg(feature = "xz")]
            LeveledCodec::Xz(level) => {
                let mut encoder = xz::write::XzEncoder::new(Vec::new(), level.compression_level());
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                Ok(compressed)
            }
            #[cfg(not(feature = "xz"))]
            LeveledCodec::Xz(_) => Err(ArrowError::ParseError(
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
        // Fast modes and the library default are representable, not just `1..=22`.
        assert_eq!(ZstdLevel::try_new(-1).unwrap().compression_level(), -1);
        assert_eq!(ZstdLevel::try_new(0).unwrap().compression_level(), 0);
        assert_eq!(ZstdLevel::try_new(1).unwrap().compression_level(), 1);
        assert_eq!(ZstdLevel::try_new(22).unwrap().compression_level(), 22);
        assert!(ZstdLevel::try_new(23).is_err());
        assert!(ZstdLevel::try_new(-131_073).is_err());
    }

    /// The published ranges are what `try_new` enforces, and every default is inside its own
    /// range — `Default` only debug-asserts that, so a test has to check it.
    #[test]
    fn level_ranges_are_published_and_contain_their_default() {
        assert_eq!((DeflateLevel::MIN, DeflateLevel::MAX), (0, 9));
        assert_eq!((ZstdLevel::MIN, ZstdLevel::MAX), (-131_072, 22));
        assert_eq!((Bzip2Level::MIN, Bzip2Level::MAX), (1, 9));
        assert_eq!((XzLevel::MIN, XzLevel::MAX), (0, 9));

        assert_eq!(
            DeflateLevel::try_new(DeflateLevel::MIN)
                .unwrap()
                .compression_level(),
            DeflateLevel::MIN
        );
        assert_eq!(
            ZstdLevel::try_new(ZstdLevel::MAX)
                .unwrap()
                .compression_level(),
            ZstdLevel::MAX
        );

        assert!(DeflateLevel::try_new(DeflateLevel::default().compression_level()).is_ok());
        assert!(ZstdLevel::try_new(ZstdLevel::default().compression_level()).is_ok());
        assert!(Bzip2Level::try_new(Bzip2Level::default().compression_level()).is_ok());
        assert!(XzLevel::try_new(XzLevel::default().compression_level()).is_ok());
    }

    /// `zstd::Encoder::new` rejects levels outside the crate's own range, so ours has to match it
    /// rather than drift from it.
    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_level_range_matches_the_crate() {
        let crate_range = zstd::compression_level_range();
        assert_eq!(
            ZstdLevel::MIN,
            *crate_range.start(),
            "zstd range is {crate_range:?}"
        );
        assert_eq!(
            ZstdLevel::MAX,
            *crate_range.end(),
            "zstd range is {crate_range:?}"
        );
        for level in [*crate_range.start(), -1, 0, *crate_range.end()] {
            assert!(
                ZstdLevel::try_new(level).is_ok(),
                "level {level} is inside {crate_range:?}"
            );
        }
    }

    #[test]
    fn bzip2_level_range() {
        assert_eq!(Bzip2Level::default().compression_level(), 6);
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

    /// `bzip2::Compression::new` panics outside `1..=9`, so the validation has to agree with it.
    #[cfg(feature = "bzip2")]
    #[test]
    fn bzip2_level_range_matches_the_crate() {
        for level in 0..=10 {
            assert_eq!(
                Bzip2Level::try_new(level).is_ok(),
                bzip2::Compression::try_new(level).is_some(),
                "level {level}"
            );
        }
    }

    /// `xz::write::XzEncoder::new` unwraps the preset error, so an out-of-range level panics
    /// inside the encoder; the validation has to agree with liblzma's presets.
    #[cfg(feature = "xz")]
    #[test]
    fn xz_level_range_matches_the_crate() {
        for level in 0..=11 {
            let encoder_accepts =
                xz::stream::Stream::new_easy_encoder(level, xz::stream::Check::Crc64).is_ok();
            assert_eq!(
                XzLevel::try_new(level).is_ok(),
                encoder_accepts,
                "level {level}"
            );
        }
    }

    /// Compressible-but-not-trivial payload, so the level actually matters.
    #[cfg(any(
        feature = "deflate",
        feature = "zstd",
        feature = "bzip2",
        feature = "xz"
    ))]
    fn sample() -> Vec<u8> {
        (0..4096).flat_map(|i: u32| i.to_le_bytes()).collect()
    }

    // Each codec's default level has to reproduce the exact bytes the writer produced before
    // levels were configurable. The comparison is against the crate's own default rather than a
    // stored blob, because the encoders' output changes between library versions.

    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_default_level_matches_the_previous_encoder() {
        let data = sample();
        let mut previous =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        previous.write_all(&data).unwrap();
        let previous = previous.finish().unwrap();
        assert_eq!(
            LeveledCodec::Deflate(DeflateLevel::default())
                .compress(&data)
                .unwrap(),
            previous
        );
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_default_level_matches_the_previous_encoder() {
        let data = sample();
        // Level 0 is what the writer passed before levels were configurable.
        let mut previous = zstd::Encoder::new(Vec::new(), 0).unwrap();
        previous.write_all(&data).unwrap();
        let previous = previous.finish().unwrap();
        assert_eq!(
            LeveledCodec::ZStandard(ZstdLevel::default())
                .compress(&data)
                .unwrap(),
            previous
        );
    }

    #[cfg(feature = "bzip2")]
    #[test]
    fn bzip2_default_level_matches_the_previous_encoder() {
        let data = sample();
        let mut previous = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        previous.write_all(&data).unwrap();
        let previous = previous.finish().unwrap();
        assert_eq!(
            LeveledCodec::Bzip2(Bzip2Level::default())
                .compress(&data)
                .unwrap(),
            previous
        );
    }

    #[cfg(feature = "xz")]
    #[test]
    fn xz_default_level_matches_the_previous_encoder() {
        let data = sample();
        // 6 was hardcoded in the writer before levels were configurable.
        let mut previous = xz::write::XzEncoder::new(Vec::new(), 6);
        previous.write_all(&data).unwrap();
        let previous = previous.finish().unwrap();
        assert_eq!(
            LeveledCodec::Xz(XzLevel::default())
                .compress(&data)
                .unwrap(),
            previous
        );
    }

    /// Every block has to be a complete deflate stream: readers decode each block on its own, and
    /// a strict one (Java's `InflaterInputStream`) expects the final-block marker rather than
    /// running into the end of the input.
    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_blocks_are_terminated_streams() {
        let data = sample();
        let codec = LeveledCodec::Deflate(DeflateLevel::default());

        for block in 0..2 {
            let compressed = codec.compress(&data).expect("compress");
            let mut decompress = flate2::Decompress::new(false);
            // `decompress_vec` writes into the vector's spare capacity, so give it room for the
            // whole block up front.
            let mut out = Vec::with_capacity(data.len() + 64);
            let status = decompress
                .decompress_vec(&compressed, &mut out, flate2::FlushDecompress::Finish)
                .expect("decompress");
            assert_eq!(
                status,
                flate2::Status::StreamEnd,
                "block {block} is not a complete deflate stream"
            );
            assert_eq!(out, data, "block {block} round-trip");
        }
    }

    /// Higher deflate levels should produce no larger output than lower levels on redundant data,
    /// and round-tripping at any level must yield identical bytes.
    #[cfg(feature = "deflate")]
    #[test]
    fn deflate_levels_produce_smaller_output_at_higher_setting() {
        // Sequential little-endian integers, which deflate compresses well.
        let data: Vec<u8> = (0..4096).flat_map(|i: u32| i.to_le_bytes()).collect();

        let fast = LeveledCodec::Deflate(DeflateLevel::try_new(1).unwrap())
            .compress(&data)
            .unwrap();
        let best = LeveledCodec::Deflate(DeflateLevel::try_new(9).unwrap())
            .compress(&data)
            .unwrap();

        assert!(
            best.len() <= fast.len(),
            "level 9 ({} bytes) should not exceed level 1 ({} bytes)",
            best.len(),
            fast.len()
        );

        for raw in [&fast, &best] {
            let round_trip = CompressionCodec::Deflate.decompress(raw).unwrap();
            assert_eq!(round_trip, data);
        }
    }
}
