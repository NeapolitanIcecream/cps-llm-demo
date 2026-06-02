use thiserror::Error;

#[derive(Debug, Error)]
pub enum DemoError {
    #[error("model protocol error: {0}")]
    ModelProtocol(String),

    #[error("OPENAI_API_KEY is required. Set it via env or --api-key.")]
    MissingApiKey,
}

pub fn model_protocol_error(message: impl Into<String>) -> DemoError {
    DemoError::ModelProtocol(message.into())
}
