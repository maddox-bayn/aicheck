# AICheck

AICheck assesses practical Python AI engineering through **20 sequential checkpoints with 10 exercises each**. Start with checkpoint 01. Each exercise has a problem statement, the required interface, a complete Python reference implementation and an executable behavioral check. A checkpoint passes when all ten exercises pass.

## Start on Linux Mint

Use Python 3.12 on Linux, macOS or WSL. From the project root, the first checkpoint requires only NumPy.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5
python start.py 1
python check.py 1 --list
```

Open `submissions/cp01/ex01.py`. Implement `unit_rows(x)` using the description at the top of the file. Predict the output for a zero vector and for `[3, 4]`, then run the exercise check.

```bash
python check.py 1 --exercise 1 --submission submissions --verbose
python check.py 1 --submission submissions
```

The first command grades only exercise 01.1. The second grades all ten and reports unfinished exercises individually. Running `start.py` again preserves existing work. Once you finish a checkpoint, compare your implementation against its Markdown file and explain one failure mode to a reviewer.

## Install the complete CPU environment

Use the active virtual environment. The CPU PyTorch wheel keeps GPU runtime libraries out of the installation. The second command installs the versions used by the checker, including the assessment frameworks for the later checkpoints.

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python check.py --reference
```

## Assessment commands

| Task | Command |
| --- | --- |
| List all 200 exercises in order | `python check.py --list` |
| Prepare checkpoint 09 without opening its solutions | `python start.py 9` |
| Grade one learner checkpoint | `python check.py 9 --submission submissions` |
| Debug a failing exercise with a traceback | `python check.py 9 --exercise 2 --submission submissions --verbose` |
| Run the published code for comparison | `python check.py 9 --reference` |
| Write a machine-readable result | `python check.py 9 --submission submissions --json > scores.json` |

The checker returns exit status 0 when every selected exercise passes and 1 when a check fails. Each exercise has a 120-second time limit, adjustable with `--timeout`. The tests call the required interface and check outcomes, so learners can use different internal implementations. The Markdown code block is the executable reference source. Learner grading loads the corresponding submission file.

## Curriculum

| Pillar | Checkpoint | Practical assessment |
| --- | --- | --- |
| 1 · Data foundations | [01 · Advanced NumPy](checkpoints/01.md) | Normalize, whiten and merge batches without retaining raw rows. |
| 1 · Data foundations | [02 · Validated data pipelines](checkpoints/02.md) | Quarantine malformed records and produce point-in-time joins. |
| 1 · Data foundations | [03 · PyTorch core](checkpoints/03.md) | Differentiate tensor computations and batch variable-length samples. |
| 2 · Classical ML | [04 · Scikit-learn pipelines](checkpoints/04.md) | Fit transforms inside folds and evaluate calibration and group leakage. |
| 2 · Classical ML | [05 · Optimization from scratch](checkpoints/05.md) | Check analytic gradients, implement Adam and backpropagate through an MLP. |
| 3 · Deep learning and vision | [06 · MLPs and CNNs](checkpoints/06.md) | Train image classifiers, freeze encoders and restore best weights. |
| 3 · Deep learning and vision | [07 · Embeddings and projection](checkpoints/07.md) | Compare sparse and dense retrieval, then quantify PCA and t-SNE distortion. |
| 4 · NLP and Transformers | [08 · Tokenization](checkpoints/08.md) | Train BPE, audit Unicode offsets and prepare causal labels. |
| 4 · NLP and Transformers | [09 · Transformer mechanics](checkpoints/09.md) | Implement masked multi-head attention and incremental KV state. |
| 4 · NLP and Transformers | [10 · Open-source model workflows](checkpoints/10.md) | Fine-tune a tiny BERT model and export a local snapshot. |
| 5 · LLM orchestration | [11 · API integration](checkpoints/11.md) | Handle streaming, call IDs, bounded retries and tool-call rounds. |
| 5 · LLM orchestration | [12 · Prompt automation](checkpoints/12.md) | Build few-shot prompts, enforce schemas and run regression cases. |
| 6 · RAG systems | [13 · Vector databases](checkpoints/13.md) | Index and query FAISS and Chroma, and construct scoped Pinecone calls. |
| 6 · RAG systems | [14 · Enterprise retrieval](checkpoints/14.md) | Hydrate parents, fuse ranks, rerank and filter evidence before generation. |
| 7 · Agent systems | [15 · Tool execution](checkpoints/15.md) | Validate, authorize and bound tool execution with audit events. |
| 7 · Agent systems | [16 · Multi-agent coordination](checkpoints/16.md) | Run an AutoGen critique team and persist LangGraph human-review decisions. |
| 8 · Performance and deployment | [17 · LLM evaluation](checkpoints/17.md) | Use Ragas and DeepEval, test judge position bias and bootstrap paired changes. |
| 8 · Performance and deployment | [18 · Inference optimization](checkpoints/18.md) | Quantize weights, merge LoRA, size KV memory and configure GPU serving. |
| 8 · Performance and deployment | [19 · Async AI serving](checkpoints/19.md) | Exercise actual FastAPI routes, streaming responses, queues and overload responses. |
| 8 · Performance and deployment | [20 · Secure AI operations](checkpoints/20.md) | Reject privileged messages and prevent unauthorized reads through semantic caches. |

## Repository layout

| Path | Purpose |
| --- | --- |
| `checkpoints/01.md` through `checkpoints/20.md` | Full exercise specifications and standalone reference code |
| `checks/01.py` through `checks/20.py` | Published, numbered behavioral checks executed per exercise |
| `start.py` | Generates learner files with required interfaces and problem descriptions |
| `check.py` | Runs the chosen exercises and emits text or JSON scores |
| `submissions/cpNN/exNN.py` | Learner implementation created by the starter |
| `.github/workflows/check.yml` | Grades every reference and each committed submission checkpoint on pushes and pull requests |

## Assessment boundaries

The CPU suite runs actual NumPy, pandas, PyTorch, scikit-learn, tokenizer, Transformers, FAISS, Chroma, LangGraph, AutoGen, Ragas, DeepEval and FastAPI code on small fixtures. Deterministic clients supply API and judge responses. Pinecone tests inspect client calls without a managed index. The vLLM exercise validates launch arguments. The Torch-TensorRT exercise verifies backend selection and warm-up using a test double. Live provider quality, CUDA engine compatibility and hardware throughput require separate infrastructure.

The grader executes trusted Python in the current process. Use an isolated execution environment before accepting unknown third-party submissions. Public checks give learners reproducible feedback. For assessment, ask learners to explain a design choice and demonstrate a new boundary case. The final security checkpoint combines application-level authorization with scoped retrieval and output handling. Its rate limiter applies within one process, and its PII patterns cover the formats listed in that exercise.

## Framework references

- Tokenizer construction, encodings and model fine-tuning. [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) · [tiktoken](https://github.com/openai/tiktoken) · [Transformers training](https://huggingface.co/docs/transformers/training)
- Persistent human review and multi-agent teams. [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [AutoGen teams](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html)
- Evaluation metrics and framework conventions. [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) · [DeepEval answer relevancy](https://deepeval.com/docs/metrics-answer-relevancy)
- Serving and security. [vLLM documentation](https://docs.vllm.ai/) · [Torch-TensorRT compilation](https://docs.pytorch.org/TensorRT/user_guide/compilation/torch_compile.html) · [OWASP prompt injection prevention](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html)

## Publish to the empty GitHub repository

If you received this project as `aicheck.zip`, extract it and work in the directory containing this README. To publish to the empty repository from an authenticated Git CLI, run the following once.

```bash
git init -b main
git add .
git commit -m "Add 20 AI engineering checkpoints and automated grading"
git remote add origin https://github.com/maddox-bayn/aicheck.git
git push -u origin main
```

For an existing clone, copy the project files into the clone, commit and push using its configured remote.
