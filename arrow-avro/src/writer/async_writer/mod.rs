// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

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
//!   [`DEFAULT_OCF_BLOCK_SIZE`] by default; tune via [`WriterBuilder::with_block_size`]. An OCF
//!   writer must be finished with [`AsyncWriter::finish`] (or [`AsyncWriter::into_inner`], which
//!   flushes and hands the sink back), because rows staged in an in-progress block are lost when
//!   the writer is dropped.
//! - **Feature-gated**: Requires the `async` feature

use crate::compression::LeveledCodec;
use crate::errors::AvroError;
use crate::writer::encoder::{RecordEncoder, write_long};
use crate::writer::format::{AvroFormat, AvroOcfFormat, AvroSoeFormat};
use crate::writer::{BlockBuf, WriteStats, WriterBuilder, sink_failed_error};
use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use bytes::Bytes;
use futures::future::{BoxFuture, FutureExt};
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::io::{AsyncWrite, AsyncWriteExt};

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
/// For custom sinks, such as remote storage via the [`object_store`] crate, implement this
/// trait directly, typically by pairing a store handle with an object path and delegating
/// [`Self::write`] / [`Self::complete`] / [`Self::abort`] to a multipart upload (e.g.
/// `object_store::buffered::BufWriter`).
///
/// [`object_store`]: https://crates.io/crates/object_store
///
/// # Example: implementing `AsyncFileWriter` for the `object_store` crate
///
/// ```no_run
/// # use std::sync::Arc;
/// use arrow_avro::writer::AsyncFileWriter;
/// use arrow_avro::errors::AvroError;
/// use bytes::Bytes;
/// use futures::FutureExt;
/// use futures::future::BoxFuture;
/// use object_store::ObjectStore;
/// use object_store::buffered::BufWriter;
/// use object_store::path::Path;
/// use tokio::io::AsyncWriteExt;
///
/// #[derive(Debug)]
/// struct ObjectStoreWriter {
///     w: BufWriter,
/// }
///
/// impl ObjectStoreWriter {
///     fn new(store: Arc<dyn ObjectStore>, path: Path) -> Self {
///         Self { w: BufWriter::new(store, path) }
///     }
/// }
///
/// impl AsyncFileWriter for ObjectStoreWriter {
///     fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), AvroError>> {
///         async move {
///             self.w.put(bs).await.map_err(|e| {
///                 AvroError::External(Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
///             })
///         }
///         .boxed()
///     }
///
///     fn complete(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
///         async move {
///             self.w.shutdown().await.map_err(|e| {
///                 AvroError::IoError(format!("Error finishing object store upload: {e}"), e)
///             })
///         }
///         .boxed()
///     }
///
///     fn abort(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
///         async move {
///             self.w.abort().await.map_err(|e| {
///                 AvroError::External(Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
///             })
///         }
///         .boxed()
///     }
/// }
/// ```
pub trait AsyncFileWriter: Send {
    /// Write the provided bytes to the underlying writer.
    ///
    /// This method may be called multiple times during the writing process.
    /// Each call provides a chunk of the Avro output that should be written
    /// to the destination. Implementations are expected to append all bytes
    /// or return an error.
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), AvroError>>;

    /// Flush any buffered data and finish the writing process.
    ///
    /// This method should ensure all data is persisted to the underlying storage.
    /// After `complete` returns `Ok(())`, the caller SHOULD NOT call `write` again.
    fn complete(&mut self) -> BoxFuture<'_, Result<(), AvroError>>;

    /// Abandon any in-flight writes and clean up partially uploaded state.
    ///
    /// Implementations backed by multipart uploads (e.g. an `object_store`-based
    /// sink, see the example on the trait documentation) should send the upstream
    /// abort here so partial uploads are not billed or left lingering.
    /// Implementations backed by local sinks (files, in-memory buffers) typically
    /// have nothing to undo and may leave the default no-op.
    ///
    /// `abort` and [`complete`](Self::complete) are mutually exclusive: callers
    /// should invoke at most one of them on a given writer. After `complete` the only thing
    /// left for `abort` to do is release the sink, so implementations may treat it as a no-op.
    fn abort(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
        async move { Ok(()) }.boxed()
    }
}

impl AsyncFileWriter for Box<dyn AsyncFileWriter + '_> {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), AvroError>> {
        self.as_mut().write(bs)
    }

    fn complete(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
        self.as_mut().complete()
    }

    fn abort(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
        self.as_mut().abort()
    }
}

impl<T: AsyncWrite + Unpin + Send> AsyncFileWriter for T {
    fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), AvroError>> {
        async move {
            self.write_all(&bs)
                .await
                .map_err(|e| AvroError::IoError(format!("Error writing bytes: {e}"), e))
        }
        .boxed()
    }

    fn complete(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
        async move {
            self.flush()
                .await
                .map_err(|e| AvroError::IoError(format!("Error flushing: {e}"), e))?;
            self.shutdown()
                .await
                .map_err(|e| AvroError::IoError(format!("Error closing: {e}"), e))
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
/// # Buffering and `finish`
///
/// An OCF writer stages encoded rows until the block reaches
/// [`DEFAULT_OCF_BLOCK_SIZE`](crate::writer::DEFAULT_OCF_BLOCK_SIZE) (or the size set by
/// [`WriterBuilder::with_block_size`]), so [`write`](Self::write) does not necessarily put
/// anything on the sink. Call [`finish`](Self::finish) when you are done, or
/// [`into_inner`](Self::into_inner) if you also need the sink back, and [`abort`](Self::abort) on
/// the error path: a writer that is dropped instead loses the block still staged.
///
/// Stream formats ([`AsyncAvroStreamWriter`]) write each row straight to the sink and only need
/// `finish` to complete it.
///
/// Construct via the convenience constructors below or
/// [`WriterBuilder::build_async`](crate::writer::WriterBuilder::build_async).
#[derive(Debug)]
pub struct AsyncWriter<W: AsyncFileWriter, F: AvroFormat> {
    writer: W,
    schema: Arc<Schema>,
    /// Format type, kept for the `AsyncWriter<W, F>` API; the format's state is consumed when
    /// the header is written and its sync marker is cached in `sync`.
    format: PhantomData<F>,
    compression: Option<LeveledCodec>,
    encoder: RecordEncoder,
    /// Sync marker of a block-framed format (`F::IS_BLOCK_FRAMED`), checked when the writer is
    /// built. `None` for stream formats, which write each record straight to the sink.
    sync: Option<[u8; 16]>,
    /// Rows staged for the next OCF block. Untouched by stream formats.
    block: BlockBuf,
    /// Rows already flushed to the sink. [`AsyncWriter::stats`] adds the rows still staged in
    /// `block` so the reported total never depends on the block boundaries.
    rows_written: u64,
    /// Number of OCF blocks emitted to the sink. Always 0 for non-OCF formats.
    blocks_written: u64,
    /// Set once a sink write fails: the partially written block cannot be repaired, so the
    /// writer refuses to touch the sink again.
    failed: bool,
    /// Set once [`complete`](AsyncFileWriter::complete) succeeded, so a second [`finish`] does
    /// not complete the sink twice.
    finished: bool,
}

/// Alias for async **Object Container File** writer.
pub type AsyncAvroWriter<W> = AsyncWriter<W, AvroOcfFormat>;

/// Alias for async **Single Object Encoding** stream writer.
pub type AsyncAvroStreamWriter<W> = AsyncWriter<W, AvroSoeFormat>;

impl<W: AsyncFileWriter> AsyncAvroWriter<W> {
    /// Create a new async Avro OCF writer with default settings.
    ///
    /// Equivalent to `WriterBuilder::new(schema).build_async::<W, AvroOcfFormat>(writer).await`.
    pub async fn new(writer: W, schema: Schema) -> Result<Self, AvroError> {
        WriterBuilder::new(schema)
            .build_async::<W, AvroOcfFormat>(writer)
            .await
    }

    /// Return a reference to the 16-byte sync marker generated for this file.
    pub fn sync_marker(&self) -> Option<&[u8; 16]> {
        self.sync.as_ref()
    }
}

impl<W: AsyncFileWriter> AsyncAvroStreamWriter<W> {
    /// Create a new async Single Object Encoding stream writer with default settings.
    pub async fn new(writer: W, schema: Schema) -> Result<Self, AvroError> {
        WriterBuilder::new(schema)
            .build_async::<W, AvroSoeFormat>(writer)
            .await
    }
}

impl<W: AsyncFileWriter, F: AvroFormat> AsyncWriter<W, F> {
    /// Constructor used by [`WriterBuilder::build_async`], which validates the schema, the
    /// encoder and the sync marker first.
    pub(crate) fn from_parts(
        writer: W,
        schema: Arc<Schema>,
        sync: Option<[u8; 16]>,
        compression: Option<LeveledCodec>,
        block_size: usize,
        capacity: usize,
        encoder: RecordEncoder,
    ) -> Self {
        Self {
            writer,
            schema,
            format: PhantomData,
            sync,
            compression,
            encoder,
            // Stream formats never stage rows, so they get no block buffer at all.
            block: BlockBuf::new(
                if F::IS_BLOCK_FRAMED { capacity } else { 0 },
                if F::IS_BLOCK_FRAMED { block_size } else { 0 },
            ),
            rows_written: 0,
            blocks_written: 0,
            failed: false,
            finished: false,
        }
    }

    /// Returns the Arrow schema (with `avro.schema` metadata) this writer was built with.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Write a single [`RecordBatch`].
    ///
    /// A batch with no rows stages nothing, so an OCF writer emits no block for it and
    /// [`stats`](Self::stats) stays put.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch schema differs from the writer's, if a sink write already
    /// failed, or once [`finish`](Self::finish) has completed the sink.
    pub async fn write(&mut self, batch: &RecordBatch) -> Result<(), AvroError> {
        self.check_failed()?;
        if self.finished {
            return Err(AvroError::InvalidArgument(
                "writer has already been finished".to_string(),
            ));
        }
        if batch.schema().fields() != self.schema.fields() {
            return Err(AvroError::SchemaError(
                "Schema of RecordBatch differs from Writer schema".to_string(),
            ));
        }

        match self.sync {
            Some(sync) => self.encode_into_block(batch, &sync).await,
            None => self.write_stream(batch).await,
        }
    }

    /// Write multiple batches.
    pub async fn write_batches(&mut self, batches: &[&RecordBatch]) -> Result<(), AvroError> {
        for batch in batches {
            self.write(batch).await?;
        }
        Ok(())
    }

    /// Emit the rows staged for the current OCF block, without finishing the writer.
    ///
    /// [`write`](Self::write) only reaches the sink when a block fills, so a long-lived writer
    /// calls this to make its rows visible — or to bound what it buffers — between writes.
    /// [`finish`](Self::finish) flushes once more and then completes the sink; stream formats
    /// have nothing to flush and this is a no-op for them.
    ///
    /// # Errors
    ///
    /// Returns an error if a sink write already failed, or once [`finish`](Self::finish) has
    /// completed the sink.
    pub async fn flush(&mut self) -> Result<(), AvroError> {
        self.check_failed()?;
        if self.finished {
            return Err(AvroError::InvalidArgument(
                "writer has already been finished".to_string(),
            ));
        }
        if let Some(sync) = self.sync {
            self.flush_block(&sync).await?;
        }
        Ok(())
    }

    /// Bytes staged for the current OCF block, waiting for [`flush`](Self::flush) or a write that
    /// fills it. Always `0` for stream formats.
    pub fn in_progress_size(&self) -> usize {
        self.block.bytes().len()
    }

    /// Rows staged for the current OCF block. Always `0` for stream formats.
    pub fn in_progress_rows(&self) -> usize {
        self.block.rows()
    }

    /// Flush any pending OCF block and signal end of writing on the underlying sink.
    ///
    /// The sink is completed once: a second call is a no-op, so the counters stay readable with
    /// [`stats`](Self::stats) afterwards. Rows staged in an in-progress block are lost if the
    /// writer is dropped without calling this method or [`abort`](Self::abort).
    pub async fn finish(&mut self) -> Result<(), AvroError> {
        self.check_failed()?;
        if self.finished {
            return Ok(());
        }
        if let Some(sync) = self.sync {
            self.flush_block(&sync).await?;
        }
        if let Err(e) = self.writer.complete().await {
            self.failed = true;
            return Err(e);
        }
        self.finished = true;
        Ok(())
    }

    /// Sink writes are not recoverable: a failed write may have delivered part of a block, so a
    /// retry would duplicate its header or payload. Only [`abort`](Self::abort) can follow.
    fn check_failed(&self) -> Result<(), AvroError> {
        if self.failed {
            return Err(sink_failed_error());
        }
        Ok(())
    }

    /// Push bytes to the sink, marking the writer unusable if the sink fails.
    async fn sink_write(&mut self, bytes: Bytes) -> Result<(), AvroError> {
        if let Err(e) = self.writer.write(bytes).await {
            self.failed = true;
            return Err(e);
        }
        Ok(())
    }

    /// Abandon the writer, dropping any in-progress OCF block and asking the underlying
    /// [`AsyncFileWriter`] to clean up partially written state (e.g. abort an in-flight
    /// multipart upload).
    ///
    /// Use this on the error path instead of [`finish`](Self::finish): the staged block is
    /// dropped instead of flushed, and `complete` is never invoked. Bytes already handed to the
    /// sink stay there, including the OCF header written by
    /// [`WriterBuilder::build_async`], so this only
    /// removes output when the sink can discard it. The default
    /// [`AsyncFileWriter::abort`] is a no-op, which means plain [`AsyncWrite`] sinks (files,
    /// in-memory buffers) keep whatever they were given and a header-only file can be left
    /// behind; multipart uploads are the case this exists for.
    ///
    /// It is also the only remaining option once a sink error has made the writer unusable, and
    /// a no-op once [`finish`](Self::finish) has completed the sink. Calling it consumes the
    /// writer.
    pub async fn abort(mut self) -> Result<(), AvroError> {
        if self.finished {
            // Nothing left to undo: `finish` already completed the sink.
            return Ok(());
        }
        // Drop any staged rows without flushing — the sink should not see a
        // partial block.
        self.block.clear();
        self.writer.abort().await
    }

    /// Snapshot of the writer's counters, queryable at any time.
    ///
    /// The counters keep reporting what was accepted and flushed after a sink error, which is
    /// terminal for the writer (see [`write`](Self::write)).
    pub fn stats(&self) -> WriteStats {
        WriteStats {
            rows_accepted: self.rows_written + self.block.rows() as u64,
            blocks_written: self.blocks_written,
        }
    }

    /// Flush any pending OCF block, complete the underlying sink, and return it.
    ///
    /// This performs the same work as [`finish`](Self::finish), so rows staged in an in-progress
    /// block are emitted instead of being discarded with the writer, and the sink is completed
    /// before it is handed back.
    ///
    /// # Errors
    ///
    /// Returns an error, and drops the sink, if the pending block could not be written or the
    /// sink could not be completed. Use [`abort`](Self::abort) when the goal is to clean up a
    /// partial upload rather than to keep it.
    pub async fn into_inner(mut self) -> Result<W, AvroError> {
        self.finish().await?;
        Ok(self.writer)
    }

    async fn encode_into_block(
        &mut self,
        batch: &RecordBatch,
        sync: &[u8; 16],
    ) -> Result<(), AvroError> {
        self.block.stage(&self.encoder, batch)?;
        if self.block.is_full() {
            self.flush_block(sync).await?;
        }
        Ok(())
    }

    /// Emit the currently accumulated OCF block. No-op if no rows are accumulated.
    ///
    /// A compressed payload goes out as three writes (header, payload, sync marker) so it can
    /// move into the sink without an intermediate memcpy; an uncompressed one is copied anyway
    /// and goes out as a single write.
    ///
    /// The sync writer's `flush_block` does the same work against a `std::io::Write`; keep the two
    /// in step.
    async fn flush_block(&mut self, sync: &[u8; 16]) -> Result<(), AvroError> {
        if self.block.is_empty() {
            return Ok(());
        }
        // The block is cleared only once all of it reached the sink. A failure keeps the staged
        // rows, so `stats` still counts them, but poisons the writer: a half-written block cannot
        // be retried, and `abort` is all that is left to call (see `check_failed`).
        let compressed = match self.compression {
            Some(codec) => Some(codec.compress(self.block.bytes())?),
            None => None,
        };
        let payload_len = compressed
            .as_deref()
            .unwrap_or_else(|| self.block.bytes())
            .len();

        let mut header_buf = Vec::<u8>::with_capacity(16);
        write_long(&mut header_buf, self.block.rows() as i64)?;
        write_long(&mut header_buf, payload_len as i64)?;
        self.sink_write(Bytes::from(header_buf)).await?;

        // An uncompressed payload is already a copy of the staging buffer, so the trailing sync
        // marker rides along in the same write. A compressed payload is not copied again just to
        // append 16 bytes, so its marker goes out on its own.
        match compressed {
            Some(compressed) => {
                self.sink_write(Bytes::from(compressed)).await?;
                self.sink_write(Bytes::copy_from_slice(sync)).await?;
            }
            None => {
                let mut block = Vec::with_capacity(self.block.bytes().len() + sync.len());
                block.extend_from_slice(self.block.bytes());
                block.extend_from_slice(sync);
                self.sink_write(Bytes::from(block)).await?;
            }
        }
        self.rows_written += self.block.rows() as u64;
        self.block.clear();
        self.blocks_written += 1;
        Ok(())
    }

    async fn write_stream(&mut self, batch: &RecordBatch) -> Result<(), AvroError> {
        let mut buf = Vec::<u8>::new();
        self.encoder.encode(&mut buf, batch)?;
        self.sink_write(Bytes::from(buf)).await?;
        self.rows_written += batch.num_rows() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::ReaderBuilder;
    use crate::writer::format::AvroOcfFormat;
    use arrow_array::{ArrayRef, BinaryArray, Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc as SArc, Mutex};

    /// The sink the writer tests share: it records the bytes it was given, counts what was done to
    /// it, and can be armed to fail after `n` more writes.
    ///
    /// Every field is shared with clones, so a test can keep one handle to watch while the writer
    /// owns another.
    #[derive(Clone, Default)]
    struct TestSink {
        written: SArc<Mutex<Vec<u8>>>,
        writes: SArc<AtomicUsize>,
        completes: SArc<AtomicUsize>,
        aborts: SArc<AtomicUsize>,
        /// `None` accepts every write; `Some(n)` accepts `n` more and then fails.
        remaining: SArc<Mutex<Option<usize>>>,
    }

    impl TestSink {
        /// Number of bytes handed to the sink so far.
        fn len(&self) -> usize {
            self.written.lock().unwrap().len()
        }

        /// Copy of everything the sink was given.
        fn bytes(&self) -> Vec<u8> {
            self.written.lock().unwrap().clone()
        }

        fn writes(&self) -> usize {
            self.writes.load(Ordering::SeqCst)
        }

        fn completes(&self) -> usize {
            self.completes.load(Ordering::SeqCst)
        }

        fn aborts(&self) -> usize {
            self.aborts.load(Ordering::SeqCst)
        }

        /// Let `writes` more writes through, then fail every subsequent one.
        fn fail_after(&self, writes: usize) {
            *self.remaining.lock().unwrap() = Some(writes);
        }

        /// Stop failing.
        fn heal(&self) {
            *self.remaining.lock().unwrap() = None;
        }
    }

    impl AsyncFileWriter for TestSink {
        fn write(&mut self, bs: Bytes) -> BoxFuture<'_, Result<(), AvroError>> {
            self.writes.fetch_add(1, Ordering::SeqCst);
            let fail = {
                let mut remaining = self.remaining.lock().unwrap();
                match *remaining {
                    Some(0) => true,
                    Some(n) => {
                        *remaining = Some(n - 1);
                        false
                    }
                    None => false,
                }
            };
            let written = self.written.clone();
            async move {
                if fail {
                    let e = std::io::Error::other("sink failure");
                    return Err(AvroError::IoError(e.to_string(), e));
                }
                written.lock().unwrap().extend_from_slice(&bs);
                Ok(())
            }
            .boxed()
        }

        fn complete(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
            self.completes.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        }

        fn abort(&mut self) -> BoxFuture<'_, Result<(), AvroError>> {
            self.aborts.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        }
    }

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
        use crate::writer::DEFAULT_OCF_BLOCK_SIZE;

        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![42])) as ArrayRef],
        )?;

        let mut buffer = Vec::new();
        let hint = 4 * DEFAULT_OCF_BLOCK_SIZE;
        let mut writer = WriterBuilder::new(schema.clone())
            .with_capacity(hint)
            .build_async::<_, AvroOcfFormat>(&mut buffer)
            .await?;
        // `build_async` forwards the builder's hints to the same staging buffer the sync writer
        // uses, so a capacity above one block has to show up there.
        assert!(
            writer.block.capacity() >= hint,
            "builder capacity not propagated, got {}",
            writer.block.capacity()
        );

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
        let recovered = writer.into_inner().await?;
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
            .with_compression(Some(crate::compression::CompressionCodec::Deflate))
            .build_async::<_, AvroOcfFormat>(&mut buffer)
            .await?;

        writer.write(&batch).await?;
        writer.finish().await?;

        let mut reader = ReaderBuilder::new().build(Cursor::new(buffer))?;
        let out = reader.next().unwrap()?;
        assert_eq!(out.num_rows(), 3);

        Ok(())
    }

    /// Snappy takes a different branch inside `compress` than the streaming codecs, and it is the
    /// codec most Avro files in the wild use, so the async writer's compressed path (three writes
    /// per block) gets covered with it too, across enough rows to split the input.
    #[tokio::test(flavor = "current_thread")]
    #[cfg(feature = "snappy")]
    async fn test_async_writer_with_snappy_compression() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from_iter_values(0..4096)) as ArrayRef],
        )?;

        let sink = TestSink::default();
        let mut writer = WriterBuilder::new(schema)
            .with_compression(Some(crate::compression::CompressionCodec::Snappy))
            .without_block_buffering()
            .build_async::<_, AvroOcfFormat>(sink.clone())
            .await?;
        let after_header = sink.writes();

        writer.write(&batch).await?;
        writer.finish().await?;

        // The compressed path still frames the block itself rather than copying the payload.
        assert_eq!(
            sink.writes() - after_header,
            3,
            "a compressed block is header, payload and marker"
        );

        let bytes = writer.into_inner().await?.bytes();
        assert!(
            bytes.len() < batch.get_array_memory_size(),
            "snappy should have shrunk the data, got {} bytes",
            bytes.len()
        );

        let reader = ReaderBuilder::new().build(Cursor::new(bytes))?;
        let mut rows = 0;
        let mut last = None;
        for batch in reader {
            let batch = batch?;
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            rows += batch.num_rows();
            last = column.iter().next_back().flatten().or(last);
        }
        assert_eq!(rows, 4096, "every row comes back");
        assert_eq!(last, Some(4095), "in order");

        Ok(())
    }

    /// Number of data blocks in an OCF container, found by walking the container rather than by
    /// scanning for sync-marker bytes, which a block payload could contain.
    ///
    /// Also checks the framing: the header's metadata map, and a sync marker after every block.
    fn count_ocf_blocks(buffer: &[u8], sync: &[u8; 16]) -> usize {
        fn read_long(buffer: &[u8], cursor: &mut usize) -> i64 {
            let mut zigzag = 0i64;
            let mut shift = 0;
            loop {
                let byte = buffer[*cursor];
                *cursor += 1;
                zigzag |= ((byte & 0x7f) as i64) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            (zigzag >> 1) ^ -(zigzag & 1)
        }

        fn read_bytes<'a>(buffer: &'a [u8], cursor: &mut usize) -> &'a [u8] {
            let len = read_long(buffer, cursor) as usize;
            let bytes = &buffer[*cursor..*cursor + len];
            *cursor += len;
            bytes
        }

        // Header: magic, `avro.schema` + `avro.codec` map, sync marker.
        assert_eq!(&buffer[..4], b"Obj\x01", "OCF magic");
        let mut cursor = 4;
        loop {
            let entries = read_long(buffer, &mut cursor);
            if entries == 0 {
                break;
            }
            for _ in 0..entries {
                read_bytes(buffer, &mut cursor); // key
                read_bytes(buffer, &mut cursor); // value
            }
        }
        assert_eq!(&buffer[cursor..cursor + 16], sync, "header sync marker");
        cursor += 16;

        // Blocks: row count, byte count, payload, sync marker.
        let mut blocks = 0;
        while cursor < buffer.len() {
            read_long(buffer, &mut cursor); // rows
            let len = read_long(buffer, &mut cursor) as usize;
            cursor += len;
            assert_eq!(
                &buffer[cursor..cursor + 16],
                sync,
                "sync marker after block {blocks}"
            );
            cursor += 16;
            blocks += 1;
        }
        blocks
    }

    /// Many small batches at the default 64 KiB block size coalesce into a single OCF block.
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
        let bytes = writer.into_inner().await.unwrap();

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

    /// A payload that contains the sync-marker bytes must still count as one block: counting
    /// marker occurrences produced a false positive here.
    #[tokio::test(flavor = "current_thread")]
    async fn test_block_counting_ignores_sync_marker_bytes_in_the_payload()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("v", DataType::Binary, false)]);
        let mut writer = AsyncAvroWriter::new(Vec::new(), schema.clone()).await?;
        let sync = *writer.sync_marker().expect("OCF sync marker");

        // Write the marker itself as the value, so the block payload contains it verbatim.
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(BinaryArray::from_vec(vec![sync.as_slice()])) as ArrayRef],
        )?;
        writer.write(&batch).await?;
        writer.finish().await?;
        let bytes = writer.into_inner().await.unwrap();

        assert_eq!(count_ocf_blocks(&bytes, &sync), 1);
        Ok(())
    }

    /// `without_block_buffering()` flushes each batch as its own block, which is what callers
    /// wanting a hard bound on block size (e.g. for decoder latency while streaming) ask for.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_without_block_buffering_flushes_per_batch()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;

        let buffer = Vec::new();
        let mut writer = WriterBuilder::new(schema)
            .without_block_buffering()
            .build_async::<_, AvroOcfFormat>(buffer)
            .await?;
        for _ in 0..4 {
            writer.write(&batch).await?;
        }
        writer.finish().await?;
        let sync = *writer.sync_marker().expect("OCF sync marker");
        let bytes = writer.into_inner().await.unwrap();

        assert_eq!(
            count_ocf_blocks(&bytes, &sync),
            4,
            "each batch should be its own OCF block"
        );

        Ok(())
    }

    /// If the encoder fails to build, no bytes should reach the sink: the header is only
    /// written once the encoder exists.
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

    /// `finish()` reports rows + blocks; `stats()` agrees mid-stream.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_finish_returns_stats() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        // 4 batches × 3 rows = 12 rows, and without buffering every batch becomes its own
        // block, so blocks_written should equal 4.
        let batches: Vec<RecordBatch> = (0..4)
            .map(|i| {
                RecordBatch::try_new(
                    Arc::new(schema.clone()),
                    vec![Arc::new(Int64Array::from(vec![i, i + 1, i + 2])) as ArrayRef],
                )
                .unwrap()
            })
            .collect();

        let mut buffer = Vec::new();
        let mut writer = WriterBuilder::new(schema)
            .without_block_buffering()
            .build_async::<_, AvroOcfFormat>(&mut buffer)
            .await?;
        for (i, b) in batches.iter().enumerate() {
            writer.write(b).await?;
            let mid = writer.stats();
            assert_eq!(mid.rows_accepted, 3 * (i as u64 + 1));
            assert_eq!(mid.blocks_written, i as u64 + 1);
        }
        writer.finish().await?;
        let stats = writer.stats();
        assert_eq!(stats.rows_accepted, 12);
        assert_eq!(stats.blocks_written, 4);
        Ok(())
    }

    /// The writer has to be `Send` for the obvious use — moving it into `tokio::spawn` — which no
    /// test that runs on a `current_thread` runtime can show.
    #[test]
    fn test_async_writer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AsyncAvroWriter<Vec<u8>>>();
        assert_send::<AsyncAvroStreamWriter<Vec<u8>>>();
    }

    /// `flush()` puts the staged block on the sink while the writer stays usable, which is what a
    /// long-lived writer needs to make rows visible or bound what it holds.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_flush_emits_the_pending_block()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;

        let mut writer = AsyncAvroWriter::new(Vec::new(), schema).await?;

        writer.write(&batch).await?;
        assert!(writer.in_progress_rows() > 0, "the batch is staged");
        assert!(writer.in_progress_size() > 0, "the block has bytes");
        assert_eq!(writer.stats().blocks_written, 0);

        writer.flush().await?;
        assert_eq!(writer.in_progress_rows(), 0, "flush emptied the block");
        assert_eq!(writer.in_progress_size(), 0);
        assert_eq!(writer.stats().blocks_written, 1);

        // Still usable: the next batch becomes its own block.
        writer.write(&batch).await?;
        assert_eq!(writer.stats().blocks_written, 1);
        writer.flush().await?;
        assert_eq!(writer.stats().blocks_written, 2);

        writer.finish().await?;
        let reader = ReaderBuilder::new().build(Cursor::new(writer.into_inner().await?))?;
        let mut rows = 0;
        for batch in reader {
            rows += batch?.num_rows();
        }
        assert_eq!(rows, 6, "both blocks read back");

        Ok(())
    }

    /// `flush()` is not a way around the finished state: the sink has been completed, so it
    /// reports the same error `write` does.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_flush_after_finish_errors() -> Result<(), Box<dyn std::error::Error>>
    {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let mut writer = AsyncAvroWriter::new(Vec::new(), schema).await?;
        writer.finish().await?;

        let err = writer.flush().await.unwrap_err();
        assert!(
            err.to_string().contains("already been finished"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    /// `into_inner()` flushes the staged block before handing the sink back, so a caller that
    /// needs the bytes cannot lose the last block.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_into_inner_flushes_staged_rows()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;

        let mut writer = AsyncAvroWriter::new(Vec::new(), schema).await?;
        writer.write(&batch).await?;
        assert_eq!(
            writer.stats().blocks_written,
            0,
            "rows should still be staged"
        );

        let buffer = writer.into_inner().await?;
        let mut reader = ReaderBuilder::new().build(Cursor::new(buffer))?;
        assert_eq!(reader.next().unwrap()?.num_rows(), 3);
        Ok(())
    }

    /// A sink failure part-way through a block poisons the writer: the block is not counted, and
    /// later calls error instead of writing the block header a second time.
    ///
    /// Compression is on so a block reaches the sink in three parts (header, payload, marker):
    /// arming the sink for one write leaves a header behind and fails the payload, which is the
    /// partial block that makes retrying unsafe. An uncompressed block is a single write, so it
    /// either lands whole or not at all.
    #[cfg(feature = "deflate")]
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_sink_failure_poisons_the_writer()
    -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;

        let sink = TestSink::default();
        let mut writer = WriterBuilder::new(schema)
            .with_compression(Some(crate::compression::CompressionCodec::Deflate))
            .without_block_buffering() // flush on every write
            .build_async::<_, AvroOcfFormat>(sink.clone())
            .await?;
        let header_len = sink.len();

        // Let the block header through, then fail on the payload.
        sink.fail_after(1);
        assert!(
            writer.write(&batch).await.is_err(),
            "the payload write should fail"
        );
        let after_failure = sink.len();
        assert!(
            after_failure > header_len,
            "the block header was written before the failure"
        );

        let failed = writer.stats();
        assert_eq!(failed.blocks_written, 0, "a failed block must not count");
        assert_eq!(
            failed.rows_accepted,
            batch.num_rows() as u64,
            "the rows are still staged"
        );

        // Retrying cannot repair the half-written block, so nothing more reaches the sink.
        sink.heal();
        assert!(
            writer.write(&batch).await.is_err(),
            "the writer stays failed"
        );
        assert!(writer.finish().await.is_err(), "the writer stays failed");
        assert_eq!(
            sink.len(),
            after_failure,
            "no bytes may be written after a sink failure"
        );

        // `abort` is the only remaining way to release the sink.
        writer.abort().await?;
        assert_eq!(sink.aborts(), 1);
        assert_eq!(sink.len(), after_failure);
        Ok(())
    }

    /// `build_async` must propagate the builder's compression level to the writer.
    #[tokio::test(flavor = "current_thread")]
    #[cfg(feature = "deflate")]
    async fn test_async_writer_builder_deflate_level_is_applied()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::compression::{CompressionCodec, DeflateLevel};

        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from_iter_values(0..1024)) as ArrayRef],
        )?;

        let mut sizes = Vec::new();
        for level in [0u32, 9] {
            let mut writer = WriterBuilder::new(schema.clone())
                .with_compression(Some(CompressionCodec::Deflate))
                .with_deflate_level(DeflateLevel::try_new(level)?)
                .build_async::<_, AvroOcfFormat>(Vec::new())
                .await?;
            for _ in 0..16 {
                writer.write(&batch).await?;
            }
            sizes.push(writer.into_inner().await?.len());
        }

        assert!(
            sizes[1] < sizes[0],
            "level 9 ({} bytes) should be smaller than level 0 ({} bytes)",
            sizes[1],
            sizes[0]
        );
        Ok(())
    }

    /// An uncompressed block goes out as two sink writes — header, then payload carrying the
    /// sync marker — while a compressed payload keeps its own write instead of being copied.
    #[tokio::test(flavor = "current_thread")]
    async fn test_writes_per_block() -> Result<(), Box<dyn std::error::Error>> {
        async fn writes_per_block(
            compression: Option<crate::compression::CompressionCodec>,
        ) -> Result<usize, Box<dyn std::error::Error>> {
            let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
            )?;
            let sink = TestSink::default();
            let mut writer = WriterBuilder::new(schema)
                .with_compression(compression)
                .without_block_buffering() // flush on every write
                .build_async::<_, AvroOcfFormat>(sink.clone())
                .await?;
            let after_header = sink.writes();
            writer.write(&batch).await?;
            writer.finish().await?;
            Ok(sink.writes() - after_header)
        }

        assert_eq!(
            writes_per_block(None).await?,
            2,
            "header, then payload with sync"
        );
        #[cfg(feature = "deflate")]
        assert_eq!(
            writes_per_block(Some(crate::compression::CompressionCodec::Deflate)).await?,
            3,
            "the compressed payload is not copied to append the sync marker"
        );
        Ok(())
    }

    /// `finish()` completes the sink exactly once; a finished writer rejects further writes and
    /// `abort` becomes a no-op.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_finish_is_idempotent() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;
        let sink = TestSink::default();

        let mut writer = AsyncAvroWriter::new(sink.clone(), schema).await?;
        writer.write(&batch).await?;
        writer.finish().await?;
        writer.finish().await?;
        assert_eq!(sink.completes(), 1, "the sink is completed once");
        assert_eq!(writer.stats().rows_accepted, 3);

        let written = sink.len();
        let err = writer.write(&batch).await.unwrap_err();
        assert!(
            err.to_string().contains("already been finished"),
            "unexpected error: {err}"
        );
        assert_eq!(sink.len(), written, "no bytes may follow a finished writer");

        writer.abort().await?;
        assert_eq!(sink.aborts(), 0, "abort after finish has nothing to undo");
        Ok(())
    }

    /// `abort()` discards staged rows and never calls `complete()` on the sink.
    /// We verify by routing through a sink that flips a flag on `complete`
    /// (and another on `abort`), and by confirming bytes from the in-progress
    /// block are never observed downstream.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_writer_abort_calls_sink_abort() -> Result<(), Box<dyn std::error::Error>> {
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef],
        )?;
        let sink = TestSink::default();

        let mut writer = AsyncAvroWriter::new(sink.clone(), schema).await?;
        // The header is already written; capture its size so abort can be shown to add nothing.
        let header_len = sink.len();
        writer.write(&batch).await?;
        // The batch is still staged (the default block is 64 KiB); abort drops it.
        writer.abort().await?;

        assert_eq!(sink.aborts(), 1, "sink.abort() is called");
        assert_eq!(
            sink.completes(),
            0,
            "sink.complete() is not called on abort"
        );
        // The OCF header stays: `abort` only removes output when the sink knows how to discard
        // it, which this hand-rolled sink does not.
        assert_eq!(
            sink.len(),
            header_len,
            "abort must not flush the in-progress block to the sink"
        );
        Ok(())
    }
}
