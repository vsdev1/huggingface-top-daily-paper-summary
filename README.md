# huggingface-top-daily-paper-summary
AI agent that reads the content of the top daily huggingface paper for a given topic and summarizes it. In order to do that, the agent uses tools that are created in [summarize_papers.py](
https://github.com/vsdev1/huggingface-top-daily-paper-summary/blob/main/summarize_papers.py).


**Currently the topic is still hardcoded to "LLM" and only the first page is read.**

The [smolagents library](https://huggingface.co/docs/smolagents/index) is used create [tools](https://huggingface.co/docs/smolagents/tutorials/tools) which an LLM calls in order to download the first  
[Huggingface daily LLM papers](https://huggingface.co/papers?q=LLM) and create a summary of it.

You have to do the following in order to run the project:
* add an .env file with your huggingface token: ```HF_TOKEN=<your_token>```
* install the project dependencies: ```pip install -r requirements.py```
* run the python file: ```python summarize_papers.py```

## Example output where it can be seen that the agent calls the given tools
```text
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── New run ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                                                                                              │
│ Output the title and the url of the top 1 paper for the topic LLM on Hugging Face daily papers and summarize this paper by reading it. Also output the topic you searched for. All this should be included in the final answer. Please use the given tools   │
│ to get the paper id, download it and read it. Then, summarize the paper in a few sentences. Do not use any other tools or libraries.                                                                                                                         │
│                                                                                                                                                                                                                                                              │
╰─ HfApiModel - Qwen/Qwen2.5-Coder-32B-Instruct ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing this code: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  topic = "LLM"                                                                                                                                                                                                                                                 
  top_paper_ids = get_hugging_face_top_daily_paper_ids_by_topic(topic=topic, num_papers=1)                                                                                                                                                                      
  print("Top paper IDs:", top_paper_ids)                                                                                                                                                                                                                        
  paper_id = top_paper_ids[0]                                                                                                                                                                                                                                   
  download_paper_by_id(paper_id=paper_id)                                                                                                                                                                                                                       
  paper_content = read_pdf_file(file_path="paper.pdf")                                                                                                                                                                                                          
  print("Paper content:", paper_content)                                                                                                                                                                                                                        
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Found paper: LLM in a flash: Efficient Large Language Model Inference with Limited
  Memory with id: 2312.11514
Found 1 papers for topic: LLM
23
Execution logs:
Top paper IDs: ['2312.11514']
Paper content: LLM in a flash:
Efficient Large Language Model Inference with Limited Memory
Keivan Alizadeh, Iman Mirzadeh* , Dmitry Belenko* , S. Karen Khatamifard,
Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, Mehrdad Farajtabar
Apple †
Abstract
Large language models (LLMs) are central to
modern natural language processing, delivering
exceptional performance in various tasks. How-
ever, their substantial computational and mem-
ory requirements present challenges, especially
for devices with limited DRAM capacity. This
paper tackles the challenge of efficiently run-
ning LLMs that exceed the available DRAM
capacity by storing the model parameters in
flash memory, but bringing them on demand
to DRAM. Our method involves constructing
an inference cost model that takes into account
the characteristics of flash memory, guiding
us to optimize in two critical areas: reduc-
ing the volume of data transferred from flash
and reading data in larger, more contiguous
chunks. Within this hardware-informed frame-
work, we introduce two principal techniques.
First, “windowing” strategically reduces data
transfer by reusing previously activated neu-
rons, and second, “row-column bundling”, tai-
lored to the sequential data access strengths
of flash memory, increases the size of data
chunks read from flash memory. These meth-
ods collectively enable running models up to
twice the size of the available DRAM, with
up to 4x and 20x increase in inference speed
compared to naive loading approaches in CPU
and GPU, respectively. Our integration of spar-
sity awareness, context-adaptive loading, and
a hardware-oriented design paves the way for
effective inference of LLMs on devices with
limited memory.
1 Introduction
In recent years, large language models (LLMs)
have demonstrated strong performance across a
wide range of natural language tasks (Brown et al.,
2020; Chowdhery et al., 2022; Touvron et al.,
2023a; Jiang et al., 2023; Gemini Team, 2023).
* Major Contribution
† {kalizadehvahid, imirzadeh, d_belenko, skhatamifard,
minsik, cdelmundo, mrastegari, farajtabar}@apple.com
Naive
Llama 2-7B
(CPU)
Ours Naive
OPT-6.7B
(CPU)
Ours Naive
OPT-6.7B
(GPU)
Ours
100
450
700
1000
2250
3100Inference Latency (ms)
Compute Load From Flash Memory Management
Figure 1: Average inference latency for a single token
when only half of the model’s memory is available: Our
method selectively loads parameters on demand for each
token generation step. The latency represents the time
required to repeatedly load parameters from flash mem-
ory, combined with the time needed for computations.
However, the unprecedented capabilities of these
models come with substantial computational and
memory requirements for inference. LLMs can
contain hundreds of billions or even trillions of pa-
rameters, which makes them challenging to load
and run efficiently, especially on personal devices.
Currently, the standard approach is to load the en-
tire model into DRAM (Dynamic Random Access
Memory) for inference (Rajbhandari et al., 2021;
Aminabadi et al., 2022). However, this severely
limits the maximum model size that can be run.
For example, a 7 billion parameter model requires
over 14GB of memory just to load the parameters
in half-precision floating point format, exceeding
the capabilities of most personal devices such as
smartphones. While it is possible to employ tech-
niques such as quantization to reduce the model
size, still, this cannot address the main limitation
of loading the entire model into DRAM.
To address this limitation, we propose to store
the model parameters in flash memory, which is
arXiv:2312.11514v3  [cs.CL]  30 Jul 2024

Out: None
[Step 0: Duration 13.31 seconds| Input tokens: 2,349 | Output tokens: 153]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing this code: ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  title = "LLM in a flash: Efficient Large Language Model Inference with Limited Memory"                                                                                                                                                                        
  paper_id = "2312.11514"                                                                                                                                                                                                                                       
  url = f"https://arxiv.org/abs/{paper_id}"                                                                                                                                                                                                                     
  summary = (                                                                                                                                                                                                                                                   
      "This paper addresses the challenge of running large language models (LLMs) with limited DRAM by storing model parameters in flash memory. "                                                                                                              
      "The authors introduce techniques such as 'windowing' and 'row-column bundling' to reduce data transfer and increase data chunk size from flash memory, respectively. "                                                                                   
      "These methods enable running models up to twice the size of available DRAM, with significant increases in inference speed compared to naive loading methods."                                                                                            
  )                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                
  final_answer({                                                                                                                                                                                                                                                
      "title": title,                                                                                                                                                                                                                                           
      "url": url,                                                                                                                                                                                                                                               
      "summary": summary,                                                                                                                                                                                                                                       
      "topic": topic                                                                                                                                                                                                                                            
  })                                                                                                                                                                                                                                                            
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
Out - Final answer: {'title': 'LLM in a flash: Efficient Large Language Model Inference with Limited Memory', 'url': 'https://arxiv.org/abs/2312.11514', 'summary': "This paper addresses the challenge of running large language models (LLMs) with limited 
DRAM by storing model parameters in flash memory. The authors introduce techniques such as 'windowing' and 'row-column bundling' to reduce data transfer and increase data chunk size from flash memory, respectively. These methods enable running models up to
twice the size of available DRAM, with significant increases in inference speed compared to naive loading methods.", 'topic': 'LLM'}
[Step 1: Duration 12.07 seconds| Input tokens: 5,941 | Output tokens: 381]

