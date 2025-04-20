# LLM Package Examples

This directory contains examples of how to use the LLM package.

## Examples

### Simple Example

The `simple_example.py` script demonstrates basic usage of the LLM package with a single provider.

```bash
python -m cloudbrain.graph.llm.examples.simple_example
```

### Streaming Example

The `streaming_example.py` script demonstrates how to use streaming with the LLM package.

```bash
python -m cloudbrain.graph.llm.examples.streaming_example
```

### Multi-Provider Example

The `multi_provider_example.py` script demonstrates how to use multiple providers with the LLM package.

```bash
python -m cloudbrain.graph.llm.examples.multi_provider_example
```

## Requirements

To run these examples, you need to have the following environment variables set:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (for Anthropic examples)

You can set these environment variables in a `.env` file in the root directory of the project.

## Notes

- The examples use the `dotenv` package to load environment variables from a `.env` file.
- The streaming example uses a simple WebSocket mock for demonstration purposes.
- The multi-provider example checks if the Anthropic API key is available before trying to use it. 