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

//! `async` API for writing [`RecordBatch`]es to Avro files
//!
//! This module provides the async counterpart to the synchronous Avro writer.
//! Configuration is shared with the sync writer through [`crate::writer::WriterBuilder`];
//! call [`WriterBuilder::build_async`](crate::writer::WriterBuilder::build_async) to obtain
//! an [`AsyncWriter`], or use the convenience constructors on [`AsyncAvroWriter`] /
//! [`AsyncAvroStreamWriter`].
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use arrow_array::{ArrayRef, Int64Array, RecordBatch};
//! use arrow_schema::{DataType, Field, Schema};
//! use arrow_avro::writer::AsyncAvroWriter;
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
//! let batch = RecordBatch::try_new(
//!     Arc::new(schema.clone()),
//!     vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
//! )?;
//!
//! let mut buffer = Vec::new();
//! let mut writer = AsyncAvroWriter::new(&mut buffer, schema).await?;
//! writer.write(&batch).await?;
//! writer.finish().await?;
//!
//! assert!(!buffer.is_empty());
//! # Ok(()) }
//! ```
//!
//! # Features
//!
//! - **OCF format**: Write Avro Object Container Files with schema, sync markers, and optional compression
//! - **SOE format**: Write Avro Single Object Encoding streams for registry-based workflows
//! - **Flexible sinks**: Works with any `AsyncWrite + Send` type or custom [`AsyncFileWriter`] implementations
//! - **Compression**: Supports all compression codecs (Deflate, Snappy, ZStandard, etc.)
//! - **Block accumulation**: Encoded rows are batched into OCF blocks of
//!   [`DEFAULT_BLOCK_SIZE`](crate::writer::DEFAULT_BLOCK_SIZE) by default; tune via
//!   [`WriterBuilder::with_block_size`](crate::writer::WriterBuilder::with_block_size)
//! - **Feature-gated**: Requires the `async` feature

use crate::compression::CompressionCodec;
use crate::writer::WriterBuilder;
use crate::writer::encoder::{RecordEncoder, write_long};
use crate::writer::format::{AvroFormat, AvroOcfFormat, AvroSoeFormat};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};
use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use std::sync::Arc;
use tokio::io::{AsyncWrite, AsyncWriteExt};

#[cfg(feature = "object_store")]
pub mod store;
#[cfg(feature = "object_store")]
pub use store::AvroObjectWriter;

/// The asynchronous interface used by [`AsyncWriter`] to write Avro files.
///
/// This trait allows [`AsyncWriter`] to be generic over different output destinations,
/// such as files, network sockets, or in-memory buffers. It abstracts the async write
/// operations needed to produce Avro output.
///
/// # Semantics
///
/// - **[`write`](Self::write)**: Writes a chunk of bytes to the underlying sink. This may be
///   called multiple times during writing. Implementations may buffer internally (or write
///   immediately), and may implement retry logic. The method is expected to append all bytes
///   or return an error. The bytes are provided as [`Bytes`] for efficient zero-copy handling.
///
/// - **[`complete`](Self::complete)**: Signals that writing is finished. Implementations should
///   flush any buffered data and finalize the output (e.g., close file handles). After `complete`
///   returns `Ok(())`, no further `write` calls should be made.
///
/// # Provided Implementations
///
/// A blanket implementation is provided for all types implementing [`AsyncWrite`] + [`Unpin`] + [`Send`],
/// which covers common types like `tokio::fs::File`, `tokio::net::TcpStream`, and `Vec<u8>`.
///
/// For custom sinks (e.g., object stores, cloud storage), implement this trait directly.
#[cfg_attr(
    feature = "object_store",
    doc = "See [`AvroObjectWriter`] for a ready-made `object_store` implementation."
)]
#[cfg_attr(
    not(feature = "object_store"),
    doc = "See `AvroObjectWriter` (enable the `object_store` feature) for a ready-made implementation."
)]
pub trait AsyncFileWriter: Send {
    /// Write the provided bytes to the underlying writer.
    ///
    /// This method may be called multiple times during the writing process.
    /// Each call provides a chunk of the Avro output that should be written
    /// to the destination. Implementations are expected to append all bytes
    /// or return an error.
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), ArrowError>>;

    /// Flush any buffered data and finish the writing process.
    ///
    /// This method should ensure all data is persisted to the underlying storage.
    /// After `complete` returns `Ok(())`, the caller SHOULD NOT call `write` again.
    fn complete(&mut self) -> BoxFuture<'_, Result<(), ArrowError>>;
}

impl AsyncFileWriter for Box<dyn AsyncFileWriter + '_> {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), ArrowError>> {
        self.as_mut().write(bs)
    }

    fn complete(&mut self) -> BoxFuture<'_, Result<(), ArrowError>> {
        self.as_mut().complete()
    }
}

impl<T: AsyncWrite + Unpin + Send> AsyncFileWriter for T {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), ArrowError>> {
        async move {
            self.write_all(&bs)
                .await
                .map_err(|e| ArrowError::IoError(format!("Error writing bytes: {e}"), e))
        }
        .boxed()
    }

    fn complete(&mut self) -> BoxFuture<'_, Result<(), ArrowError>> {
        async move {
            self.flush()
                .await
                .map_err(|e| ArrowError::IoError(format!("Error flushing: {e}"), e))?;
            self.shutdown()
                .await
                .map_err(|e| ArrowError::IoError(format!("Error closing: {e}"), e))
        }
        .boxed()
    }
}

/// Generic async Avro writer.
///
/// This type is generic over the output async sink (`W`) and the Avro format (`F`).
/// You'll usually use the concrete aliases:
///
/// * **[`AsyncAvroWriter`]** for **OCF** (Object Container File)
/// * **[`AsyncAvroStreamWriter`]** for **SOE** Avro streams
///
/// Construct via the convenience constructors below or
/// [`WriterBuilder::build_async`](crate::writer::WriterBuilder::build_async).
pub struct AsyncWriter<W: AsyncFileWriter, F: AvroFormat> {
    writer: W,
    schema: Arc<Schema>,
    format: F,
    compression: Option<CompressionCodec>,
    block_size: usize,
    encoder: RecordEncoder,
    /// Staging buffer holding the encoded bytes of an in-progress OCF block.
    /// Always empty for non-OCF formats.
    block_buf: Vec<u8>,
    /// Number of rows currently accumulated in `block_buf`.
    block_rows: usize,
}

/// Alias for async **Object Container File** writer.
pub type AsyncAvroWriter<W> = AsyncWriter<W, AvroOcfFormat>;

/// Alias for async **Single Object Encoding** stream writer.
pub type AsyncAvroStreamWriter<W> = AsyncWriter<W, AvroSoeFormat>;

impl<W: AsyncFileWriter> AsyncAvroWriter<W> {
    /// Create a new async Avro OCF writer with default settings.
    ///
    /// Equivalent to `WriterBuilder::new(schema).build_async::<W, AvroOcfFormat>(writer).await`.
    pub async fn new(writer: W, schema: Schema) -> Result<Self, ArrowError> {
        WriterBuilder::new(schema)
            .build_async::<W, AvroOcfFormat>(writer)
            .await
    }

    /// Return a reference to the 16-byte sync marker generated for this file.
    pub fn sync_marker(&self) -> Option<&[u8; 16]> {
        self.format.sync_marker()
    }
}

impl<W: AsyncFileWriter> AsyncAvroStreamWriter<W> {
    /// Create a new async Single Object Encoding stream writer with default settings.
    pub async fn new(writer: W, schema: Schema) -> Result<Self, ArrowError> {
        WriterBuilder::new(schema)
            .build_async::<W, AvroSoeFormat>(writer)
            .await
    }
}

impl<W: AsyncFileWriter, F: AvroFormat> AsyncWriter<W, F> {
    /// Constructor used by [`WriterBuilder::build_async`].
    ///
    /// Not part of the public API: callers should configure via [`WriterBuilder`] instead,
    /// which takes care of the schema/encoder/header validation that needs to happen first.
    pub(crate) fn from_parts(
        writer: W,
        schema: Arc<Schema>,
        format: F,
        compression: Option<CompressionCodec>,
        block_size: usize,
        capacity: usize,
        encoder: RecordEncoder,
    ) -> Self {
        Self {
            writer,
            schema,
            format,
            compression,
            block_size,
            encoder,
            block_buf: Vec::with_capacity(capacity),
            block_rows: 0,
        }
    }

    /// Returns the Arrow schema (with `avro.schema` metadata) used by this writer.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Write a single [`RecordBatch`].
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        if batch.schema().fields() != self.schema.fields() {
            return Err(ArrowError::SchemaError(
                "Schema of RecordBatch differs from Writer schema".to_string(),
            ));
        }

        match self.format.sync_marker().copied() {
            Some(sync) => self.encode_into_block(batch, &sync).await,
            None => self.write_stream(batch).await,
        }
    }

    /// Write multiple batches.
    pub async fn write_batches(&mut self, batches: &[&RecordBatch]) -> Result<(), ArrowError> {
        for batch in batches {
            self.write(batch).await?;
        }
        Ok(())
    }

    /// Flush any pending OCF block and signal end of writing on the underlying sink.
    pub async fn finish(&mut self) -> Result<(), ArrowError> {
        if let Some(sync) = self.format.sync_marker().copied() {
            self.flush_block(&sync).await?;
        }
        self.writer.complete().await
    }

    /// Consume the writer and return the underlying async sink.
    ///
    /// Note: any rows still buffered in an in-progress OCF block are dropped.
    /// Call [`finish`](Self::finish) first to ensure all data is written.
    pub fn into_inner(self) -> W {
        self.writer
    }

    async fn encode_into_block(
        &mut self,
        batch: &RecordBatch,
        sync: &[u8; 16],
    ) -> Result<(), ArrowError> {
        // Encode the batch into the staging buffer; flush only when the configured
        // block size threshold is reached (or on finish()). Tiny per-batch blocks
        // compress poorly and amplify per-block framing overhead.
        self.encoder.encode(&mut self.block_buf, batch)?;
        self.block_rows += batch.num_rows();
        if self.block_buf.len() >= self.block_size {
            self.flush_block(sync).await?;
        }
        Ok(())
    }

    /// Emit the currently accumulated OCF block as three writes (header bytes,
    /// payload, sync marker) so the encoded payload can move into the sink without
    /// an intermediate memcpy. No-op if no rows are accumulated.
    async fn flush_block(&mut self, sync: &[u8; 16]) -> Result<(), ArrowError> {
        if self.block_rows == 0 {
            return Ok(());
        }
        let payload = match self.compression {
            Some(codec) => Bytes::from(codec.compress(&self.block_buf)?),
            None => Bytes::from(std::mem::take(&mut self.block_buf)),
        };
        let mut header_buf = Vec::<u8>::with_capacity(16);
        write_long(&mut header_buf, self.block_rows as i64)?;
        write_long(&mut header_buf, payload.len() as i64)?;
        self.writer.write(Bytes::from(header_buf)).await?;
        self.writer.write(payload).await?;
        self.writer.write(Bytes::copy_from_slice(sync)).await?;
        // Reset block state. If compression was applied, `block_buf` was not moved,
        // so clear it explicitly to drop the encoded source bytes.
        self.block_buf.clear();
        self.block_rows = 0;
        Ok(())
    }

    async fn write_stream(&mut self, batch: &RecordBatch) -> Result<(), ArrowError> {
        let mut buf = Vec::<u8>::new();
        self.encoder.encode(&mut buf, batch)?;
        self.writer.write(Bytes::from(buf)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::ReaderBuilder;
    use crate::writer::format::AvroOcfFormat;
    use arrow_array::{ArrayRef, Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use std::io::Cursor;

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_avro_writer_ocf() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            ],
        )?;

        let mut buffer = Vec::new();
        let mut writer = AsyncAvroWriter::new(&mut buffer, schema).await?;
        writer.write(&batch).await?;
        writer.finish().await?;

        // Read back using sync reader
        let mut reader = ReaderBuilder::new().build(Cursor::new(buffer))?;
        let out = reader.next().unwrap()?;
        assert_eq!(out.num_rows(), 3);

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_avro_stream_writer() -> Result<(), Box<dyn std::error::Error>> {
        use crate::schema::{AvroSchema, SINGLE_OBJECT_MAGIC, SchemaStore};

        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);

        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef],
        )?;

        let mut buffer = Vec::new();
        let mut writer = AsyncAvroStreamWriter::new(&mut buffer, schema.clone()).await?;
        writer.write(&batch).await?;
        writer.finish().await?;

        // Validate SOE prefix: magic bytes (0xC3, 0x01) + 8-byte fingerprint
        assert!(buffer.len() >= 10, "buffer too short for SOE prefix");
        assert_eq!(
            &buffer[0..2],
            &SINGLE_OBJECT_MAGIC,
            "SOE magic bytes mismatch"
        );

        // Round-trip decode using Decoder with SchemaStore
        let avro_schema = AvroSchema::try_from(&schema)?;
        let mut store = SchemaStore::new(); // Rabin fingerprint by default
        store.register(avro_schema)?;

        let mut decoder = ReaderBuilder::new()
            .with_writer_schema_store(store)
            .build_decoder()?;

        let consumed = decoder.decode(&buffer)?;
        assert_eq!(consumed, buffer.len(), "decoder should consume all bytes");

        let decoded = decoder.flush()?.expect("expected decoded batch");
        assert_eq!(decoded.num_rows(), 3);

        // Verify actual values match
        let col = decoded
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32Array");
        assert_eq!(col.value(0), 10);
        assert_eq!(col.value(1), 20);
        assert_eq!(col.value(2), 30);

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_multiple_batches() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);

        let batch1 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
        )?;

        let batch2 = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![3, 4])) as ArrayRef],
        )?;

        let mut buffer = Vec::new();
        let mut writer = AsyncAvroWriter::new(&mut buffer, schema).await?;
        writer.write_batches(&[&batch1, &batch2]).await?;
        writer.finish().await?;

        let reader = ReaderBuilder::new().build(Cursor::new(buffer))?;
        let mut total_rows = 0;
        for batch in reader {
            let out = batch?;
            total_rows += out.num_rows();
        }
        assert_eq!(total_rows, 4);

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_builder_configuration() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![42])) as ArrayRef],
        )?;

        let mut buffer = Vec::new();
        let mut writer = WriterBuilder::new(schema.clone())
            .with_capacity(2048)
            .build_async::<_, AvroOcfFormat>(&mut buffer)
            .await?;

        writer.write(&batch).await?;
        writer.finish().await?;

        let mut reader = ReaderBuilder::new().build(Cursor::new(buffer))?;
        let out = reader.next().unwrap()?;
        assert_eq!(out.num_rows(), 1);

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_into_inner() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int32Array::from(vec![99])) as ArrayRef],
        )?;

        // Use an owned Vec so we can call into_inner() and get it back
        let buffer = Vec::new();
        let mut writer = AsyncAvroWriter::new(buffer, schema).await?;
        writer.write(&batch).await?;
        writer.finish().await?;

        // Actually call into_inner() and verify we get the buffer back
        let recovered = writer.into_inner();
        assert!(!recovered.is_empty());

        // Verify the recovered buffer is valid Avro
        let mut reader = ReaderBuilder::new().build(Cursor::new(recovered))?;
        let out = reader.next().unwrap()?;
        assert_eq!(out.num_rows(), 1);

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_schema_mismatch_error() -> Result<(), Box<dyn std::error::Error>> {
        let schema1 = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let schema2 = Schema::new(vec![Field::new("name", DataType::Utf8, false)]);

        let batch = RecordBatch::try_new(
            Arc::new(schema2.clone()),
            vec![Arc::new(StringArray::from(vec!["test"])) as ArrayRef],
        )?;

        let mut buffer = Vec::new();
        let mut writer = AsyncAvroWriter::new(&mut buffer, schema1).await?;

        let result = writer.write(&batch).await;
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    #[cfg(feature = "deflate")]
    async fn test_async_writer_with_deflate_compression() -> Result<(), Box<dyn std::error::Error>>
    {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;

        let mut buffer = Vec::new();
        let mut writer = WriterBuilder::new(schema.clone())
            .with_compression(Some(CompressionCodec::Deflate(Default::default())))
            .build_async::<_, AvroOcfFormat>(&mut buffer)
            .await?;

        writer.write(&batch).await?;
        writer.finish().await?;

        let mut reader = ReaderBuilder::new().build(Cursor::new(buffer))?;
        let out = reader.next().unwrap()?;
        assert_eq!(out.num_rows(), 3);

        Ok(())
    }

    /// Count the number of OCF blocks in a buffer by counting trailing sync
    /// markers (one per block). The sync marker also appears once in the header,
    /// so subtract that occurrence.
    fn count_ocf_blocks(buffer: &[u8], sync: &[u8; 16]) -> usize {
        let mut count = 0usize;
        let mut i = 0usize;
        while i + 16 <= buffer.len() {
            if &buffer[i..i + 16] == sync {
                count += 1;
                i += 16;
            } else {
                i += 1;
            }
        }
        // Subtract 1 for the sync marker that lives in the file header itself.
        count.saturating_sub(1)
    }

    /// Many small batches at the default 64 KiB block size should be coalesced
    /// into a single OCF block, instead of one block per batch as the previous
    /// implementation produced.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_accumulates_small_batches_into_one_block()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batches: Vec<RecordBatch> = (0..32)
            .map(|i| {
                RecordBatch::try_new(
                    Arc::new(schema.clone()),
                    vec![Arc::new(Int64Array::from(vec![i, i + 1])) as ArrayRef],
                )
                .unwrap()
            })
            .collect();

        let buffer = Vec::new();
        let mut writer = AsyncAvroWriter::new(buffer, schema).await?;
        for b in &batches {
            writer.write(b).await?;
        }
        writer.finish().await?;
        let sync = *writer.sync_marker().expect("OCF sync marker");
        let bytes = writer.into_inner();

        assert_eq!(
            count_ocf_blocks(&bytes, &sync),
            1,
            "expected accumulation into a single OCF block"
        );

        // Also verify the data round-trips correctly.
        let reader = ReaderBuilder::new().build(Cursor::new(bytes))?;
        let total: usize = reader
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(total, 64);

        Ok(())
    }

    /// `block_size = 0` restores the legacy per-batch flush behavior, which is
    /// useful for callers that need to bound block size from above (e.g. for
    /// decoder latency in streaming scenarios).
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_block_size_zero_flushes_per_batch()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;

        let buffer = Vec::new();
        let mut writer = WriterBuilder::new(schema)
            .with_block_size(0)
            .build_async::<_, AvroOcfFormat>(buffer)
            .await?;
        for _ in 0..4 {
            writer.write(&batch).await?;
        }
        writer.finish().await?;
        let sync = *writer.sync_marker().expect("OCF sync marker");
        let bytes = writer.into_inner();

        assert_eq!(
            count_ocf_blocks(&bytes, &sync),
            4,
            "block_size=0 should flush each batch as its own OCF block"
        );

        Ok(())
    }

    /// If the encoder fails to build, no bytes should reach the sink. This
    /// exercises the validate-encoder-before-header ordering required by
    /// review feedback on #9241.
    #[tokio::test(flavor = "current_thread")]
    async fn test_build_async_validates_encoder_before_writing_header()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::collections::HashMap;
        let mut md = HashMap::new();
        md.insert(
            crate::schema::SCHEMA_METADATA_KEY.to_string(),
            "{ this is not valid JSON".to_string(),
        );
        let schema = Schema::new_with_metadata(vec![Field::new("id", DataType::Int64, false)], md);

        let mut buffer = Vec::new();
        let result = WriterBuilder::new(schema)
            .build_async::<_, AvroOcfFormat>(&mut buffer)
            .await;
        assert!(result.is_err());
        assert!(
            buffer.is_empty(),
            "no bytes should have been written when encoder construction fails"
        );

        Ok(())
    }
}
