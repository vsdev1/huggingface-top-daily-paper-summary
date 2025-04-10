from huggingface_hub import HfApi
from smolagents import tool, CodeAgent, HfApiModel
import arxiv
from dotenv import load_dotenv, find_dotenv
import os
from pypdf import PdfReader


@tool
def get_hugging_face_top_daily_paper_ids_by_topic(topic: str, num_papers: int = 1) -> list[str]:
    """
    This is a tool that returns a list of top arxiv paper ids from the hugging face daily papers by a topic.
    It returns the list of top arxiv paper ids.

    Args:
        topic: The paper topic for which to get the ids.
        num_papers: The number of papers to return.
    Returns:
        A list of top arxiv paper ids.
    """
    api = HfApi()
    papers = api.list_papers(query=topic)
 
    papers_ids = []
    if papers:
        for i, paper in enumerate(papers):
            if i >= num_papers:
                break
            print(f"Found paper: {paper.title} with id: {paper.id}")
            papers_ids.append(paper.id)
    else:
        print(f"No papers found for topic: {topic}")    
    print(f"Found {len(papers_ids)} papers for topic: {topic}")
    return papers_ids


@tool
def download_paper_by_id(paper_id: str) -> None:
    """
    This tool gets the id of a paper and downloads it from arxiv. It saves the paper locally 
    in the current directory as "paper.pdf".

    Args:
        paper_id: The id of the paper to download.
    """
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
    paper.download_pdf(filename="paper.pdf")
    return None


@tool
def read_pdf_file(file_path: str) -> str:
    """
    This function reads the first page of a PDF file and returns its content as a string.
    Args:
        file_path: The path to the PDF file.
    Returns:
        A string containing the content of the PDF file.
    """
    content = ""
    reader = PdfReader('paper.pdf')
    print(len(reader.pages))
    pages = reader.pages[:1]
    for page in pages:
        content += page.extract_text()
    return content


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())

    model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

    model = HfApiModel(model_id=model_id)
    agent = CodeAgent(tools=[get_hugging_face_top_daily_paper_ids_by_topic,
                            download_paper_by_id,
                            read_pdf_file],
                    model=model,
                    add_base_tools=False,
                    additional_authorized_imports=["dotenv", "os", "arxiv", "pypdf"]
                    )

    agent.run(
        ("Output the title and the url of the top 1 paper for the topic LLM on Hugging Face daily papers and summarize this paper by reading it. "
        "Also output the topic you searched for. All this should be included in the final answer. "
        "Please use the given tools to get the paper id, download it and read it. Then, summarize the paper in a few sentences. Do not use any other tools or libraries."
        )
    )