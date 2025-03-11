# LLMs.txt Generator

A Python tool to generate LLMs.txt and LLMs-full.txt files from documentation URLs using Firecrawl APIs. These files are formatted for easy ingestion and training of Large Language Models.

## Features

- **Documentation Crawling**: Discover and process linked pages within documentation sites
- **AI-Powered Summarization**: Generate concise summaries using Ollama or OpenAI models
- **Multiple Output Formats**:
  - `llms.txt`: Concise summaries of each page with links
  - `llms-full.txt`: Complete content of each page
- **Direct URL Processing**: Option to process a single URL without crawling
- **Flexible Configuration**: Customizable settings for API endpoints, models, and processing options

## Installation

1. Clone this repository:
```bash
git clone https://github.com/fede03billy/llms-txt.git
cd llms-txt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7 or higher
- A running Firecrawl instance (local or remote)
- Optional: Ollama for local summarization
- Optional: OpenAI API key for cloud-based summarization

## Usage

### Basic Usage

Generate LLMs.txt files from a documentation URL:

```bash
python llms_text_gen.py "https://example.com/docs/overview"
```

### Direct Processing (Recommended)

Process a specific URL without crawling (useful if crawling is problematic):

```bash
python llms_text_gen.py "https://example.com/docs/overview" --skip-crawl
```

### Using AI for Summaries

#### With Ollama (Local)

```bash
python llms_text_gen.py "https://example.com/docs/overview" --summarizer ollama --model "phi3:mini" --skip-crawl
```

#### With OpenAI (Cloud)

```bash
python llms_text_gen.py "https://example.com/docs/overview" --summarizer openai --openai-api-key "your_key" --model "gpt-4o-mini" --skip-crawl
```

### Advanced Options

```bash
python llms_text_gen.py "https://example.com/docs/overview" \
  --api-key "your_firecrawl_key" \
  --base-url "http://firecrawl-server.com" \
  --port 3002 \
  --api-version "v1" \
  --max-urls 200 \
  --output-dir "custom_output" \
  --summarizer ollama \
  --ollama-url "http://ollama-server:11434" \
  --model "gemma2:2b"
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `url` | Documentation URL to process | (Required) |
| `--api-key` | Firecrawl API key | None |
| `--base-url` | Base URL for Firecrawl instance | http://localhost:3002 |
| `--port` | Port number for Firecrawl instance | (From base-url) |
| `--api-version` | API version to use | v1 |
| `--max-urls` | Maximum number of URLs to process | 100 |
| `--output-dir` | Directory to save output files | output |
| `--skip-crawl` | Skip crawling and process the provided URL directly | False |
| `--summarizer` | Summarization method (heuristic, ollama, openai) | heuristic |
| `--ollama-url` | Base URL for Ollama API | http://localhost:11434 |
| `--openai-api-key` | OpenAI API key | None |
| `--model` | Model name for summarization | phi3:mini |

## Output

The tool generates two files:

1. **llms.txt** - Concise overview with summaries:
```
# example.com llms.txt

- [Overview Page](https://example.com/docs/overview): This page provides an introduction to the API and its key features.
- [Getting Started](https://example.com/docs/getting-started): Step-by-step guide to setting up and making your first API call.
```

2. **llms-full.txt** - Complete content of each page:
```
# example.com llms-full.txt

## Overview Page
This documentation provides comprehensive information about our API service...

Source: https://example.com/docs/overview

## Getting Started
To begin using our API, you need to first create an account...

Source: https://example.com/docs/getting-started
```

## Troubleshooting

- If you encounter issues with crawling, try using the `--skip-crawl` option
- Ensure your Firecrawl instance is running and accessible
- Check the Firecrawl API documentation for any changes to endpoint parameters
- When using Ollama, ensure it's running and accessible at the specified URL

## License

[MIT License](LICENSE)