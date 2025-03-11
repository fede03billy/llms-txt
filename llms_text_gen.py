import argparse
import os
import time
import json
import re
import requests
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, List, Optional


class CustomLLMsTextGenerator:
    """
    A class to create LLMs.txt files from documentation URLs using Firecrawl's APIs.
    Generates both concise and full versions of documentation content.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "http://localhost:3002",
        api_version: str = "v1",
        summarizer: str = "heuristic",
        openai_api_key: str = None,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "phi3:mini",
    ):
        """
        Initialize the CustomLLMsTextGenerator.

        Args:
            api_key: Your Firecrawl API key (optional for self-hosted instances)
            base_url: Base URL for your local Firecrawl instance (default: http://localhost:3002)
            api_version: API version to use (default: "v1")
            summarizer: Summarization method to use ('heuristic', 'ollama', or 'openai')
            openai_api_key: OpenAI API key for summarization (if using 'openai')
            ollama_base_url: Base URL for Ollama API (if using 'ollama')
            model_name: Model name to use for summarization (if using 'ollama' or 'openai')
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.headers = {}
        if api_key:
            self.headers = {"Authorization": f"Bearer {api_key}"}

        # Summarization settings
        self.summarizer = summarizer
        self.openai_api_key = openai_api_key
        self.ollama_base_url = ollama_base_url
        self.model_name = model_name

    def generate_llms_text(self, url: str, max_urls: int = 100) -> Dict[str, Any]:
        """
        Generate LLMs.txt content from a documentation URL by crawling and scraping.

        Args:
            url: The documentation URL to process
            max_urls: Maximum number of URLs to process (default: 100)

        Returns:
            Dictionary containing generated LLMs.txt content
        """
        # Step 1: Crawl the website to discover URLs
        crawl_results = self.crawl_website(url, max_urls)

        if not crawl_results.get("success", False):
            return {
                "success": False,
                "error": f"Crawl failed: {crawl_results.get('error', 'Unknown error')}",
            }

        # Extract URLs from crawl results
        data = crawl_results.get("data", [])
        print(f"Crawl results data type: {type(data)}")

        # Extract URLs from the data
        urls_to_scrape = set()  # Use a set to avoid duplicates
        base_url = "/".join(url.split("/")[:3])  # Get the base URL (scheme + domain)

        # Extract URLs from markdown content
        if isinstance(data, list):
            print("Detected list format response")
            for item in data:
                if isinstance(item, dict):
                    # Try to extract markdown content
                    if "markdown" in item:
                        markdown_content = item["markdown"]
                        # Extract URLs using regex pattern for markdown links
                        links = re.findall(r"\[.*?\]\((.*?)\)", markdown_content)
                        for link in links:
                            # Make sure we have absolute URLs
                            if not link.startswith(("http://", "https://")):
                                link = urljoin(base_url, link)
                            urls_to_scrape.add(link)

                    # Also check if there are URLs in metadata
                    if "metadata" in item and isinstance(item["metadata"], dict):
                        metadata = item["metadata"]
                        # Check for URL fields in metadata
                        for key, value in metadata.items():
                            if isinstance(value, str) and value.startswith(
                                ("http://", "https://")
                            ):
                                if (
                                    key == "url"
                                    or key == "sourceURL"
                                    or "link" in key.lower()
                                ):
                                    urls_to_scrape.add(value)

                        # If there's a specific URL field, prioritize it
                        if "url" in metadata and isinstance(metadata["url"], str):
                            urls_to_scrape.add(metadata["url"])

                # If the item itself is a string URL
                elif isinstance(item, str) and item.startswith(("http://", "https://")):
                    urls_to_scrape.add(item)

        # If we still don't have URLs, try to extract from the original URL
        if not urls_to_scrape:
            print("Fallback to original URL")
            urls_to_scrape.add(url)

            # Extract common documentation paths from the base URL
            doc_paths = [
                "/docs",
                "/documentation",
                "/guide",
                "/api",
                "/reference",
                "/quickstart",
                "/getting-started",
                "/tutorial",
            ]

            for path in doc_paths:
                urls_to_scrape.add(urljoin(base_url, path))

        # Create a list from the set of unique URLs
        urls_to_scrape = list(urls_to_scrape)

        if not urls_to_scrape:
            print("Could not extract URLs from response. Response data:")
            print(data)
            return {
                "success": False,
                "error": "Could not extract URLs from crawl results",
            }

        print(f"Found {len(urls_to_scrape)} URLs to scrape:")
        for url_to_scrape in urls_to_scrape[:10]:  # Show first 10 for debugging
            print(f"  - {url_to_scrape}")
        if len(urls_to_scrape) > 10:
            print(f"  ... and {len(urls_to_scrape) - 10} more")

        # Filter URLs to only include those from the same domain and path
        filtered_urls = self._filter_urls(urls_to_scrape, url)

        if not filtered_urls:
            print("No URLs left after filtering. Using original URL as fallback.")
            filtered_urls = [url]

        print(f"Filtered to {len(filtered_urls)} relevant documentation URLs")

        # Step 2: Scrape content from these URLs
        scrape_results = self.scrape_urls(filtered_urls)

        if not scrape_results.get("success", False):
            return {
                "success": False,
                "error": f"Scrape failed: {scrape_results.get('error', 'Unknown error')}",
            }

        # Step 3: Process scraped content into LLMs.txt format
        scraped_data = scrape_results.get("data", [])
        if not scraped_data:
            return {"success": False, "error": "No content found in scrape results"}

        # Generate LLMs.txt content
        llms_txt = self._generate_llms_txt_content(scraped_data)
        llms_full_txt = self._generate_llms_full_txt_content(scraped_data)

        return {
            "success": True,
            "data": {"llmstxt": llms_txt, "llmsfulltxt": llms_full_txt},
            "status": "completed",
        }

    def _filter_urls(self, urls: List[str], base_url: str) -> List[str]:
        """
        Filter URLs to only include those from the same domain and path.

        Args:
            urls: List of URLs to filter
            base_url: The base URL to filter against

        Returns:
            Filtered list of URLs
        """
        parsed_base_url = urlparse(base_url)
        base_path = parsed_base_url.path.rstrip("/")

        filtered_urls = []
        for url_to_check in urls:
            parsed_url_to_check = urlparse(url_to_check)

            # Skip URLs that are not part of the same domain
            if parsed_base_url.netloc != parsed_url_to_check.netloc:
                continue

            # Check if this URL is part of the documentation
            if base_path and not parsed_url_to_check.path.startswith(base_path):
                # Special case: if we're looking at /docs/overview, accept any /docs/ URL
                if not (
                    base_path.endswith("/overview")
                    and parsed_url_to_check.path.startswith(base_path[:-9])
                ):
                    continue

            filtered_urls.append(url_to_check)

        return filtered_urls

    def crawl_website(self, url: str, max_urls: int = 100) -> Dict[str, Any]:
        """
        Crawl a website using Firecrawl's crawl API.

        Args:
            url: The website URL to crawl
            max_urls: Maximum number of URLs to crawl (default: 100)

        Returns:
            Dictionary containing crawl results
        """
        endpoint = f"{self.base_url}/{self.api_version}/crawl"

        # Use 'limit' parameter as per API documentation
        data = {"url": url, "limit": max_urls}

        try:
            print(f"Crawling {url} (max URLs: {max_urls})...")
            print(f"POST request to: {endpoint}")
            print(f"Headers: {self.headers}")
            print(f"Data: {data}")

            response = requests.post(endpoint, headers=self.headers, json=data)
            response.raise_for_status()

            job_data = response.json()
            job_id = job_data.get("id")

            if not job_id:
                return {"success": False, "error": "No job ID returned"}

            print(f"Started crawl job with ID: {job_id}")

            return self._poll_for_crawl_results(job_id)

        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP Error: {e}\nResponse: {e.response.text if hasattr(e, 'response') else 'No response'}",
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request Error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected Error: {e}"}

    def _poll_for_crawl_results(
        self, job_id: str, max_attempts: int = 60, interval: int = 5
    ) -> Dict[str, Any]:
        """
        Poll for crawl job results.

        Args:
            job_id: The crawl job ID
            max_attempts: Maximum number of polling attempts (default: 60)
            interval: Polling interval in seconds (default: 5)

        Returns:
            Dictionary containing crawl results
        """
        endpoint = f"{self.base_url}/{self.api_version}/crawl/{job_id}"

        for attempt in range(max_attempts):
            try:
                response = requests.get(endpoint, headers=self.headers)
                response.raise_for_status()

                result = response.json()

                # Check if we have a dictionary with status field
                if isinstance(result, dict) and "status" in result:
                    status = result.get("status")
                    print(
                        f"Job status: {status} (attempt {attempt + 1}/{max_attempts})"
                    )

                    if status == "completed":
                        # If the result has a data field, return it
                        # Otherwise assume the result itself is the data
                        if "data" in result:
                            return {"success": True, "data": result.get("data")}
                        else:
                            print(
                                f"Direct result format detected, data keys: {result.keys() if isinstance(result, dict) else 'not a dict'}"
                            )
                            return {"success": True, "data": result}

                    if status == "failed":
                        return {"success": False, "error": "Crawl job failed"}
                # Handle case where we get a list of results directly
                elif isinstance(result, list):
                    print(
                        f"Job status: completed (direct data, attempt {attempt + 1}/{max_attempts})"
                    )
                    return {"success": True, "data": result}
                # Handle other response structures
                else:
                    print(f"Unexpected response structure: {type(result)}")
                    if attempt == max_attempts - 1:  # on last attempt
                        return {
                            "success": True,
                            "data": result,
                        }  # try to use whatever we got

                time.sleep(interval)

            except requests.exceptions.HTTPError as e:
                return {
                    "success": False,
                    "error": f"HTTP Error while polling: {e}\nResponse: {e.response.text if hasattr(e, 'response') else 'No response'}",
                }
            except requests.exceptions.RequestException as e:
                return {"success": False, "error": f"Request Error while polling: {e}"}
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Unexpected Error while polling: {e}",
                }

        return {"success": False, "error": "Timed out waiting for crawl to complete"}

    def scrape_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Scrape content from a list of URLs using Firecrawl's scrape API.
        The API only accepts one URL at a time, so we'll make multiple requests.

        Args:
            urls: List of URLs to scrape

        Returns:
            Dictionary containing scraped content
        """
        print(f"Scraping {len(urls)} URLs...")

        all_results = []
        success_count = 0

        for i, url in enumerate(urls):
            try:
                print(f"Scraping URL {i+1}/{len(urls)}: {url}")

                # For each URL, make a separate request with the correct format
                data = {
                    "url": url,
                    "formats": ["markdown"],  # Request markdown format
                    "onlyMainContent": True,  # Only get main content
                }

                response = requests.post(
                    f"{self.base_url}/{self.api_version}/scrape",
                    headers=self.headers,
                    json=data,
                )
                response.raise_for_status()

                result = response.json()

                # Handle different response formats
                if isinstance(result, dict) and result.get("success", False):
                    # Extract the data from the result
                    content_data = result.get("data", {})

                    # Create a standardized format for the scraped content
                    scraped_content = {
                        "url": url,
                        "title": content_data.get("metadata", {}).get(
                            "title", "Untitled Page"
                        ),
                        "markdown": content_data.get("markdown", ""),
                        "metadata": content_data.get("metadata", {}),
                    }

                    all_results.append(scraped_content)
                    success_count += 1
                else:
                    print(f"  Warning: Could not extract content from {url}")

            except requests.exceptions.HTTPError as e:
                print(f"  Error scraping {url}: {e}")
                if hasattr(e, "response") and e.response:
                    print(f"  Response: {e.response.text}")
            except Exception as e:
                print(f"  Error scraping {url}: {e}")

        if success_count > 0:
            print(f"Successfully scraped {success_count}/{len(urls)} URLs")
            return {"success": True, "data": all_results}
        else:
            return {"success": False, "error": "Failed to scrape any URLs"}

    def _generate_llms_txt_content(self, scraped_data: List[Dict[str, Any]]) -> str:
        """
        Generate concise LLMs.txt content from scraped data.

        Args:
            scraped_data: List of scraped content data

        Returns:
            Formatted LLMs.txt content
        """
        # Extract domain from the first item with a URL
        domain = self._extract_domain_from_data(scraped_data)

        content = [f"# {domain} llms.txt\n"]

        # Track progress
        total_items = len(scraped_data)
        print(
            f"Generating summaries for {total_items} pages using {self.summarizer}..."
        )

        for i, item in enumerate(scraped_data):
            # Extract URL, title and text from different possible structures
            url, title, text = self._extract_content_fields(item)

            # Skip if we don't have either URL or text
            if not (url or text):
                continue

            # Clean up text
            text = re.sub(r"\s+", " ", text).strip()

            # Generate summary
            summary = self._summarize_text(text, title, url)

            # Add entry to content
            content.append(f"- [{title}]({url}): {summary}")

            # Show progress
            print(f"Summarized {i+1}/{total_items} - {title}")

        return "\n".join(content)

    def _generate_llms_full_txt_content(
        self, scraped_data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate detailed LLMs-full.txt content from scraped data.

        Args:
            scraped_data: List of scraped content data

        Returns:
            Formatted LLMs-full.txt content
        """
        # Extract domain from the first item with a URL
        domain = self._extract_domain_from_data(scraped_data)

        content = [f"# {domain} llms-full.txt\n"]

        for item in scraped_data:
            # Extract URL, title and text from different possible structures
            url, title, text = self._extract_content_fields(item)

            # Skip if we don't have either URL or text
            if not (url or text):
                continue

            # Clean up text
            text = re.sub(r"\s+", " ", text).strip()

            # Add entry to content
            content.append(f"## {title}")
            content.append(f"{text}\n")
            if url:
                content.append(f"Source: {url}\n")

        return "\n".join(content)

    def _extract_domain_from_data(self, scraped_data: List[Dict[str, Any]]) -> str:
        """Extract domain from scraped data items."""
        if not scraped_data:
            return "documentation"

        for item in scraped_data:
            if isinstance(item, dict):
                if "metadata" in item and "url" in item["metadata"]:
                    return urlparse(item["metadata"]["url"]).netloc
                elif "url" in item:
                    return urlparse(item["url"]).netloc

        return "documentation"

    def _extract_content_fields(self, item: Dict[str, Any]) -> tuple:
        """Extract URL, title and text from an item."""
        url = ""
        title = "Untitled Page"
        text = ""

        if isinstance(item, dict):
            # Try to extract URL
            if "url" in item:
                url = item["url"]
            elif "metadata" in item and "url" in item["metadata"]:
                url = item["metadata"]["url"]

            # Try to extract title
            if "title" in item:
                title = item["title"]
            elif "metadata" in item and "title" in item["metadata"]:
                title = item["metadata"]["title"]

            # Try to extract text content
            if "text" in item:
                text = item["text"]
            elif "markdown" in item:
                text = item["markdown"]
            elif "content" in item:
                text = item["content"]

        return url, title, text

    def _summarize_text(self, text: str, title: str, url: str) -> str:
        """
        Summarize text using the selected summarization method.

        Args:
            text: The text to summarize
            title: The title of the page
            url: The URL of the page

        Returns:
            Summarized text
        """
        # Use different summarization methods based on configuration
        if self.summarizer == "ollama":
            return self._summarize_with_ollama(text, title)
        elif self.summarizer == "openai":
            return self._summarize_with_openai(text, title)
        else:
            # Default to heuristic method
            return self._summarize_heuristic(text)

    def _summarize_heuristic(self, text: str) -> str:
        """
        Use a heuristic approach to summarize text.

        Args:
            text: The text to summarize

        Returns:
            Summarized text
        """
        # If text is short enough, return it as is
        if len(text) <= 250:
            return text

        # Try to extract the first paragraph that seems like a summary
        paragraphs = text.split(". ")

        # Look for a good first paragraph (at least 100 chars but not too long)
        for i, para in enumerate(paragraphs[:3]):  # Check first few paragraphs
            # Skip very short paragraphs
            if len(para) < 50:
                continue

            # If this paragraph is a good size, return it
            if 100 <= len(para) <= 300:
                return para.strip() + "."

        # Fallback: get first 250 characters
        summary = text[:250].strip()
        if len(text) > 250:
            # Try to end at a sentence boundary
            last_period = summary.rfind(".")
            if last_period > 150:  # Only cut at period if we have enough text
                summary = summary[: last_period + 1]
            else:
                summary += "..."

        return summary

    def _summarize_with_ollama(self, text: str, title: str) -> str:
        """
        Use Ollama to summarize text.

        Args:
            text: The text to summarize
            title: The title of the page

        Returns:
            Summarized text
        """
        try:
            # Prepare a truncated version of the text to prevent context overflow
            truncated_text = text[:4000] if len(text) > 4000 else text

            prompt = f"""Summarize the following documentation page titled "{title}" in 1-2 sentences (maximum 250 characters):

{truncated_text}

Keep the summary factual, informative and concise. Focus on what this page explains or teaches."""

            # Call Ollama API
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            )

            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()

                # Limit length and clean up
                if len(summary) > 250:
                    summary = summary[:247] + "..."

                return summary
            else:
                print(
                    f"Error from Ollama API: {response.status_code} - {response.text}"
                )
                return self._summarize_heuristic(text)  # Fall back to heuristic

        except Exception as e:
            print(f"Error using Ollama for summarization: {e}")
            return self._summarize_heuristic(text)  # Fall back to heuristic

    def _summarize_with_openai(self, text: str, title: str) -> str:
        """
        Use OpenAI to summarize text.

        Args:
            text: The text to summarize
            title: The title of the page

        Returns:
            Summarized text
        """
        if not self.openai_api_key:
            print(
                "OpenAI API key not provided. Falling back to heuristic summarization."
            )
            return self._summarize_heuristic(text)

        try:
            # Prepare a truncated version of the text to prevent token overflow
            truncated_text = text[:4000] if len(text) > 4000 else text

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            }

            data = {
                "model": self.model_name if self.model_name else "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You summarize documentation pages in 1-2 sentences (maximum 250 characters). Keep summaries factual, informative and concise.",
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this documentation page titled '{title}':\n\n{truncated_text}",
                    },
                ],
                "max_tokens": 100,
                "temperature": 0.3,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            )

            if response.status_code == 200:
                result = response.json()
                summary = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

                # Limit length and clean up
                if len(summary) > 250:
                    summary = summary[:247] + "..."

                return summary
            else:
                print(
                    f"Error from OpenAI API: {response.status_code} - {response.text}"
                )
                return self._summarize_heuristic(text)  # Fall back to heuristic

        except Exception as e:
            print(f"Error using OpenAI for summarization: {e}")
            return self._summarize_heuristic(text)  # Fall back to heuristic

    def _scrape_single_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single URL using the Firecrawl API.

        Args:
            url: The URL to scrape

        Returns:
            Dictionary containing the scraped content
        """
        endpoint = f"{self.base_url}/{self.api_version}/scrape"

        try:
            print(f"Scraping URL: {url}")

            data = {"url": url, "formats": ["markdown"], "onlyMainContent": True}

            response = requests.post(endpoint, headers=self.headers, json=data)
            response.raise_for_status()

            result = response.json()

            if isinstance(result, dict) and result.get("success", False):
                return {"success": True, "data": result.get("data", {})}
            else:
                return {
                    "success": False,
                    "error": "Failed to extract content from response",
                }

        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP Error: {e}\nResponse: {e.response.text if hasattr(e, 'response') else 'No response'}",
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request Error: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected Error: {e}"}

    def _generate_direct_llms_txt(self, url: str) -> str:
        """
        Generate llms.txt content directly from a single URL without crawling.

        Args:
            url: The URL to process

        Returns:
            Formatted llms.txt content
        """
        print(f"Directly processing URL: {url}")

        # Scrape the single URL
        scrape_result = self._scrape_single_url(url)

        if not scrape_result.get("success", False):
            return f"# Failed to process {url}\n\nError: {scrape_result.get('error', 'Unknown error')}"

        # Get the domain for the header
        domain = urlparse(url).netloc

        # Create the content
        content = [f"# {domain} llms.txt\n"]

        # Extract content from the scraped data
        data = scrape_result.get("data", {})
        title = data.get("metadata", {}).get("title", "Untitled Page")
        text = data.get("markdown", "")

        # Clean up text
        text = re.sub(r"\s+", " ", text).strip()

        # Generate summary
        summary = self._summarize_text(text, title, url)

        # Add entry to content
        content.append(f"- [{title}]({url}): {summary}")

        return "\n".join(content)

    def _generate_direct_llms_full_txt(self, url: str) -> str:
        """
        Generate llms-full.txt content directly from a single URL without crawling.

        Args:
            url: The URL to process

        Returns:
            Formatted llms-full.txt content
        """
        print(f"Directly processing URL: {url}")

        # Scrape the single URL
        scrape_result = self._scrape_single_url(url)

        if not scrape_result.get("success", False):
            return f"# Failed to process {url}\n\nError: {scrape_result.get('error', 'Unknown error')}"

        # Get the domain for the header
        domain = urlparse(url).netloc

        # Create the content
        content = [f"# {domain} llms-full.txt\n"]

        # Extract content from the scraped data
        data = scrape_result.get("data", {})
        title = data.get("metadata", {}).get("title", "Untitled Page")
        text = data.get("markdown", "")

        # Clean up text
        text = re.sub(r"\s+", " ", text).strip()

        # Add entry to content
        content.append(f"## {title}")
        content.append(f"{text}\n")
        content.append(f"Source: {url}\n")

        return "\n".join(content)

    def save_llms_text(
        self, results: Dict[str, Any], output_dir: str, prefix: str = ""
    ) -> None:
        """
        Save LLMs.txt content to files.

        Args:
            results: Generation results
            output_dir: Directory to save files
            prefix: Prefix for output filenames
        """
        if not results.get("success", False):
            print(f"Cannot save results: {results.get('error', 'Unknown error')}")
            return

        data = results.get("data", {})

        os.makedirs(output_dir, exist_ok=True)
        # Save llms.txt
        llms_txt = data.get("llmstxt", "")
        if llms_txt:
            filename = os.path.join(output_dir, f"{prefix}llms.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(llms_txt)
            print(f"Saved llms.txt to {filename}")

        # Save llms-full.txt
        llms_full_txt = data.get("llmsfulltxt", "")
        if llms_full_txt:
            filename = os.path.join(output_dir, f"{prefix}llms-full.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(llms_full_txt)
            print(f"Saved llms-full.txt to {filename}")


def parse_domain(url: str) -> str:
    """Extract domain from URL to use as filename prefix."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain.replace(".", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLMs.txt content from documentation URLs using Firecrawl APIs"
    )
    parser.add_argument("url", help="Documentation URL to process")
    parser.add_argument(
        "--api-key", help="Firecrawl API key (optional for self-hosted instances)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:3002",
        help="Base URL for local Firecrawl instance",
    )
    parser.add_argument(
        "--port", type=int, help="Port number (overrides port in base-url if specified)"
    )
    parser.add_argument("--api-version", default="v1", help="API version to use")
    parser.add_argument(
        "--max-urls", type=int, default=100, help="Maximum number of URLs to process"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip crawling and just use the provided URL",
    )

    # Summarization options
    parser.add_argument(
        "--summarizer",
        choices=["heuristic", "ollama", "openai"],
        default="heuristic",
        help="Summarization method to use (default: heuristic)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for Ollama API (default: http://localhost:11434)",
    )
    parser.add_argument("--openai-api-key", help="OpenAI API key for summarization")
    parser.add_argument(
        "--model",
        default="phi3:mini",
        help="Model name to use for summarization (default: phi3:mini for Ollama, gpt-4o-mini for OpenAI)",
    )

    args = parser.parse_args()

    # If port is specified, update the base_url
    if args.port:
        parsed_url = urlparse(args.base_url)
        netloc = parsed_url.netloc.split(":")[0]  # Get hostname without port
        scheme = parsed_url.scheme
        base_url = f"{scheme}://{netloc}:{args.port}"
        print(f"Using custom port {args.port}, updated base URL: {base_url}")
    else:
        base_url = args.base_url

    # Validate summarizer choice
    if args.summarizer == "openai" and not args.openai_api_key:
        print(
            "Warning: OpenAI summarizer selected but no API key provided. Falling back to heuristic summarization."
        )
        args.summarizer = "heuristic"

    # Create the generator with appropriate settings
    generator = CustomLLMsTextGenerator(
        api_key=args.api_key,
        base_url=base_url,
        api_version=args.api_version,
        summarizer=args.summarizer,
        openai_api_key=args.openai_api_key,
        ollama_base_url=args.ollama_url,
        model_name=args.model,
    )

    print(f"Generating LLMs.txt content from {args.url}")
    print(f"Max URLs to process: {args.max_urls}")

    # Skip crawl if requested
    if args.skip_crawl:
        print("Skipping crawl phase and just processing the provided URL")
        results = {
            "success": True,
            "data": {
                "llmstxt": generator._generate_direct_llms_txt(args.url),
                "llmsfulltxt": generator._generate_direct_llms_full_txt(args.url),
            },
            "status": "completed",
        }
    else:
        results = generator.generate_llms_text(url=args.url, max_urls=args.max_urls)

    if results.get("success", False):
        prefix = f"{parse_domain(args.url)}_"
        generator.save_llms_text(results, args.output_dir, prefix)
        print("Generation completed successfully")
    else:
        print(f"Generation failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
