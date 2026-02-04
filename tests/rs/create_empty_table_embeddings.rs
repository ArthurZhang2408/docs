// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema};
use lancedb::{
    connect,
    database::CreateTableMode,
    embeddings::{
        sentence_transformers::SentenceTransformersEmbeddings, EmbeddingDefinition,
        EmbeddingFunction,
    },
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    let tempdir = tempfile::tempdir().unwrap();
    let uri = tempdir.path().to_str().unwrap();

    // Register an embedding function in the database's embedding registry.
    let embedding = SentenceTransformersEmbeddings::builder().build()?;
    let embedding = Arc::new(embedding);

    let db = connect(uri).execute().await?;
    db.embedding_registry()
        .register("sentence-transformers", embedding.clone())?;

    // --8<-- [start:rs_create_empty_table_with_embeddings]
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        // The vector column will be populated automatically from the embedding definition.
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384),
            false,
        ),
    ]));

    let _table = db
        .create_empty_table("my_empty_table", schema)
        .mode(CreateTableMode::Overwrite)
        .add_embedding(EmbeddingDefinition::new(
            "text",
            "sentence-transformers",
            Some("vector"),
        ))?
        .execute()
        .await?;
    // --8<-- [end:rs_create_empty_table_with_embeddings]

    Ok(())
}
