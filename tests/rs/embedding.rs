// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Embedding-related snippets for the Rust SDK.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};

use lancedb::connect;
use lancedb::connection::CreateTableMode;
use lancedb::embeddings::{EmbeddingDefinition, EmbeddingFunction, EmbeddingRegistry};
use lancedb::Result;

fn make_schema_with_embedding() -> (Arc<Schema>, EmbeddingDefinition) {
    // NOTE: We define the embedding function separately and attach it to the table
    // using `.add_embedding(...)`.
    //
    // The schema includes the source field ("text") and the vector field ("vector").
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));

    // Use a built-in embedding function from the registry.
    // This snippet is focused on table creation; the specific provider/model is not important.
    let registry = EmbeddingRegistry::new();
    let func = registry.get("sentence-transformers").unwrap().create();

    let definition = EmbeddingDefinition::new("text", "vector", func);
    (schema, definition)
}

// --8<-- [start:create_empty_table_with_embedding]
async fn create_empty_table_with_embedding(uri: &str) -> Result<()> {
    let db = connect(uri).execute().await?;

    let (schema, embedding_definition) = make_schema_with_embedding();

    let _table = db
        .create_empty_table("test", schema)
        .mode(CreateTableMode::Overwrite)
        .add_embedding(embedding_definition)
        .execute()
        .await?;

    Ok(())
}
// --8<-- [end:create_empty_table_with_embedding]

#[tokio::main]
async fn main() -> Result<()> {
    let temp_dir = tempfile::tempdir().unwrap();
    let uri = temp_dir.path().join("ex_lancedb");

    create_empty_table_with_embedding(uri.to_str().unwrap()).await?;

    // Keep the binary non-empty to avoid warnings.
    let _ = RecordBatchIterator::new(
        vec![RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["ok"]))],
        )
        .unwrap()]
        .into_iter()
        .map(Ok),
        Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)])),
    );

    Ok(())
}
