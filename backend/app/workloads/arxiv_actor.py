import requests
from urllib.parse import urlparse
import ray



def landing_to_pdf(url: str) -> str | None:
    """
    Convert a landing page URL into a direct PDF link if accessible.
    Returns the PDF URL, or None if not accessible.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    # Case 1: arXiv
    if "arxiv.org" in hostname:
        paper_id = parsed.path.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"
        try:
            if requests.head(pdf_url, timeout=5).status_code == 200:
                return pdf_url
        except requests.RequestException:
            return None
        return None

    # Case 2: PubMed
    if "pubmed.ncbi.nlm.nih.gov" in hostname:
        return None

    # Case 3: DOI.org redirects
    if "doi.org" in hostname:
        try:
            r = requests.get(url, allow_redirects=True, timeout=10)
            return landing_to_pdf(r.url)  # recurse with final landing page
        except requests.RequestException:
            return None

    # Case 4: Others (discard by default)
    return None


def transform_urls(urls: list[str]):
    ds = ray.data.from_items([{"url": url} for url in urls])

    # Apply the transformation in parallel
    ds = ds.map(lambda u: {"url": u["url"], "pdf": landing_to_pdf(u["url"])})

    # Filter out Nones
    ds = ds.filter(lambda row: row["pdf"] is not None)
    return ds


if __name__ == "__main__":
    urls = [
        "https://arxiv.org/abs/2106.01345",
        "https://pubmed.ncbi.nlm.nih.gov/1234567/",
        "https://doi.org/10.48550/arXiv.2106.01345",
    ]

    result_ds = transform_urls(urls)

    # Collect results into Python
    results = result_ds.take_all()
    print("Results:", results)
