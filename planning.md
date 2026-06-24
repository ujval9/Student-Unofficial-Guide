# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

The topic I have selected is off-campus housing for students at University at Buffalo, particularly for new and international students who are considering housing around North and South Campus.

Knowledge about this topic is very useful and relevant since it will help people stay safe, save money, and feel comfortable in their neighborhood; the choice of a bad area, landlord, or a contract can result in long commutes, problems with personal security, and expenses that cannot be reversed once signed. The information is hard to obtain through official sources, because all the essential details are stored in fragmented Facebook communities and Reddit topics, such as the noise level in particular housing complexes, affordable rent rates by bedrooms and area, which neighborhoods to avoid, ways of commuting without a car, and treatment of international students by landlords. Official website pages of UB just give generic tips on safety and leasing.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1  | r/UBreddit – "best off campus housing"                              | Thread listing popular off-campus complexes and linking to an off-campus housing superthread.                     | https://www.reddit.com/r/UBreddit/comments/158idir/best_off_campus_housing/                                        |
| 2  | r/UBreddit – "Hey incoming freshman looking for off campus housing" | Advice for new students on neighborhoods, rent expectations, and how to start the housing search.                | https://www.reddit.com/r/UBreddit/comments/1r9493z/hey_incoming_freshman_looking_for_off_campus/                   |
| 3  | r/UBreddit – "Off-Campus Housing near North Campus"                 | Discussion comparing specific North-Campus-adjacent complexes (e.g., Station, Villas, Amherst Manor).            | https://www.reddit.com/r/UBreddit/comments/vjvutj/offcampus_housing_near_north_campus/                             |
| 4  | r/UBreddit – "Popular Off Campus Housing Neighborhoods/Streets"     | Long post outlining common UB student neighborhoods and key streets to consider.                                  | https://www.reddit.com/r/UBreddit/comments/ibn2fg/popular_off_campus_housing_neighborhoodsstreets/                 |
| 5  | r/UBreddit – "Off Campus Housing; NO CAR"                           | Guidance on where to live and typical rents if you rely on UB buses and do not own a car.                         | https://www.reddit.com/r/UBreddit/comments/gu82ij/off_campus_housing_no_car/                                       |
| 6  | UB – Off-Campus Living Guide                                        | Official UB guide covering how to search, safety, leases, and responsibilities when living off campus.            | https://www.buffalo.edu/community/neighbors/students/off-campus-living-guide.html                                  |
| 7  | UB – Living Off-Campus                                              | UB page with resources and tips for students currently or planning to live off campus.                            | https://www.buffalo.edu/community/information-for-students/living-off-campus.html                                  |
| 8  | UB Off-Campus Housing Portal (OCH101)                               | UB-branded off-campus housing portal listing rentals near campus for students, faculty, and staff.                | https://buffalo.och101.com                                                                                         |
| 9  | RentCollegePads – UB Off-Campus Search                              | Student-oriented listing site filtered to University at Buffalo off-campus rentals.                               | https://www.rentcollegepads.com/off-campus-housing/university-at-buffalo/search                                    |
| 10 | Apartments.com – Houses near SUNY Buffalo North Campus              | Houses for rent near North Campus with price, bed/bath count, and amenity information.                            | https://www.apartments.com/off-campus-housing/ny/buffalo/state-university-of-new-york-buffalo-north-campus/houses/ |

## Chunking Strategy

In this particular case, a chunk size of 800–1,000 tokens with an overlap of 150–200 tokens between consecutive chunks will be used.

It comprises discussions on Reddit, guides to housing, rental listings, and housing off-campus housing resources at universities. Such documents are usually structured as relatively short posts or pages containing informative materials, and not lengthy technical manuals. In other words, a chunk size of 800–1,000 tokens will be large enough to include all the discussions, recommendations, and details concerning the housing, neighborhoods, transportation, and rent in the documents.

An overlap of 150–200 tokens will help ensure continuity when dealing with documents containing relevant information at the boundaries between chunks. This can be particularly helpful with Reddit threads as recommendations and explanations can be spread across several paragraphs. Thus, overlap ensures continuity while avoiding unnecessary duplicates in the vector database.

**Chunk size:** 250 tokens *(revised down from 800–1,000 — see note below)*

**Overlap:** 50 tokens *(revised down from 150–200, keeping the same ~20% ratio)*

**Reasoning:** The dataset contains Reddit discussions, university housing guides, and rental-related resources. I originally specified 800–1,000 tokens assuming the documents would be medium-length and context-heavy. Once implemented, the actual corpus measured only ~9,200 tokens across all 10 sources: the Reddit threads are short (a question plus a handful of comments), the UB guide pages are brief, and the three listing sites (OCH101, RentCollegePads, Apartments.com) are JavaScript apps that expose almost no static text. Token counts are measured with the `bert-base-uncased` tokenizer, which the planned embedding model `nomic-embed-text-v1.5` is built on.

**Why the change:** At 800–1,000 tokens almost every document collapsed into a single chunk (16 chunks total — below the 50-chunk healthy floor), so a query for one apartment complex couldn't be distinguished from the rest of the thread. Reducing to 250 tokens with 50-token overlap (a) isolates individual recommendations and comments as standalone, retrievable thoughts, and (b) raises the corpus to 52 chunks, within the healthy 50–2,000 range. I kept the overlap at ~20% of chunk size so a recommendation that straddles a boundary still lands intact in one chunk. The chunker is paragraph/sentence-aware (splits on natural boundaries, falling back to token windows only for oversized units), so chunks stay coherent rather than cutting mid-sentence.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 (sentence-transformers, 384-dim) *(revised from nomic-embed-text-v1.5 — see note below)*

**Top-k:** 5 *(revised from 10 — see note below)*

**Implementation note (why the change):** I originally planned `nomic-embed-text-v1.5` via the Groq embeddings API with top-k=10. During implementation I switched to **all-MiniLM-L6-v2 run locally through sentence-transformers** because it needs no API key, has no rate limits, and works fully offline — which matters for a small project iterating repeatedly on a tiny (~9k-token) corpus, and avoids spending Groq quota on embeddings. I also dropped **top-k from 10 to 5**: with only 52 chunks, k=10 pulls in loosely related material that would dilute the generation context, whereas k=5 keeps the set tight. Empirically, retrieval on all 5 evaluation queries returned on-topic chunks with cosine distances of 0.27–0.55 (all under the 0.6–0.7 weak-match threshold), so the smaller, free, local model is more than sufficient here. The Groq LLM (`llama-3.3-70b-versatile`) is still used for generation in Milestone 5.

**Production tradeoff reflection:** all-MiniLM-L6-v2 is the right call for this project, but if deploying to real users and cost were not a concern, I would evaluate larger and domain-specific embedding models (including nomic-embed-text-v1.5) that could improve retrieval accuracy on informal, student-written housing content. Key tradeoffs would include:

- **Accuracy:** Larger embedding models may better capture nuanced housing-related queries, slang, abbreviations, and implicit preferences expressed in Reddit discussions and rental listings.
- **Context Length:** Models that support longer input sequences can encode more information from lengthy housing guides and discussion threads without requiring aggressive chunking.
- **Multilingual Support:** Strong multilingual embeddings would improve search quality for international students who may mix English with other languages when searching for housing information.
- **Domain-Specific Performance:** A model fine-tuned on housing, rental, or community discussion data could provide more relevant retrieval results than a general-purpose embedding model.
- **Latency and Scalability:** Higher-quality models often require more computation, larger vector sizes, and increased storage. These factors can impact response times and infrastructure requirements at scale.
- **Generator Compatibility:** Since the retrieval pipeline feeds results into llama-3.3-70b-versatile on Groq (with a ~131k token context window), I would also consider how embedding quality affects the number of chunks that can be retrieved and passed to the LLM while maintaining acceptable latency.

I would prioritize retrieval accuracy, multilingual robustness, and compatibility with the downstream LLM while balancing storage requirements, vector dimensionality, and user-facing response times.

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

---
## Evaluation Plan

| # | Question                                                                                 | Expected Answer                                                                                                                                                  |
| - | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | As a student without a car, which neighborhoods near UB are best for off-campus housing? | The answer should mention areas such as University Heights and Amherst/North Campus areas, explain transportation options, and discuss affordability.            |
| 2 | What do students think about popular housing complexes near North Campus?                | The answer should compare at least two complexes (e.g., Station, Villas, Amherst Manor) and mention pros and cons such as cost, location, noise, and management. |
| 3 | Is a budget of $750 per month enough for off-campus housing near UB?                     | The answer should explain what type of housing is realistic within this budget and suggest places to search for listings.                                        |
| 4 | What should I check before signing an off-campus lease?                                  | The answer should mention reviewing the lease, checking utilities, visiting the property, understanding deposits, and reading reviews.                           |
| 5 | How do students commute to North Campus without a car?                                   | The answer should discuss UB Stampede buses, apartment shuttles, walking, and common challenges such as winter weather.                                          |

---

## Anticipated Challenges

1. **Conflicting information.** Different students may have very different opinions about the same apartment complex, making it difficult to provide a single clear answer.

2. **Irrelevant retrieval results.** Some Reddit posts contain unrelated discussions that may be retrieved instead of useful housing information.

3. **Chunk boundaries.** Important details may be split across multiple chunks, causing the retriever to miss useful context.

4. **Outdated information.** Rent prices and apartment availability change frequently, so some information in the dataset may become outdated.

---

## Architecture

```text
+----------------------+
| Document Ingestion   |
| (Python scripts)     |
+----------+-----------+
           |
           v
+----------------------+
| Chunking             |
| (Custom Python code) |
+----------+-----------+
           |
           v
+----------------------+
| Embeddings           |
| all-MiniLM-L6-v2     |
| (sentence-transf.)   |
+----------+-----------+
           |
           v
+----------------------+
| Vector Store         |
| ChromaDB (cosine)    |
+----------+-----------+
           |
           v
+----------------------+
| Retrieval            |
| Top-k = 5            |
+----------+-----------+
           |
           v
+----------------------+
| Generation           |
| llama-3.3-70b        |
| (Groq API)           |
+----------------------+
```

---

## AI Tool Plan

### 1. Document Ingestion

* **AI tool:** ChatGPT / Groq LLM (`llama-3.3-70b-versatile`) via Groq API.
* **Input:**

  * The “Domain” description and “Sources” table from this planning document.
  * A list of target URLs (Reddit threads, UB public guides, listing portals).
  * Constraints: respect `robots.txt`, avoid login-gated content, and output pure text with metadata (source, URL, title).
* **Expected output:**

  * Python code that crawls each URL, removes HTML and unnecessary page elements (navigation bars, advertisements, sidebars), normalizes the text, and stores records in the format `{id, source, url, title, raw_text}`.
  * Data should be saved in a local storage format such as JSONL or SQLite.
* **Verification:**

  * Run the script on a small sample of URLs.
  * Inspect saved records to confirm that the correct content is captured, metadata fields are populated correctly, and login-only pages are excluded.

### 2. Chunking

* **AI tool:** ChatGPT / Groq LLM.
* **Input:**

  * The chunking requirements defined in this planning document, including target chunk size and overlap.
  * Source-specific heuristics (shorter chunks for Reddit comments and longer chunks for guides or informational pages).
  * Example raw documents from both Reddit and UB housing resources.
* **Expected output:**

  * A Python function `chunk_text(doc_text, max_tokens, overlap_tokens)` that uses a tokenizer compatible with the embedding model.
  * Ordered chunks that preserve references to the original document and include chunk identifiers.
* **Verification:**

  * Test the function on sample documents.
  * Confirm that chunks are not empty, overlaps are applied correctly, sentences are not cut in the middle, and related information generally remains together.

### 3. Embedding + Vector Store

* **AI tool:** ChatGPT / Groq LLM.
* **Input:**

  * The embedding model selected for this project: `nomic-embed-text-v1_5` through the Groq Embeddings API.
  * The chosen vector database (ChromaDB) and schema containing chunk ID, source, URL, embedding, and text.
* **Expected output:**

  * A Python module that:

    * Generates embeddings using `client.embeddings.create(input=[chunk.text], model="nomic-embed-text-v1_5")`.
    * Processes embeddings in batches for efficiency.
    * Stores embeddings and metadata in ChromaDB.
* **Verification:**

  * Confirm that embedding requests complete successfully.
  * Verify that embedding dimensions match model specifications.
  * Run a similarity search using a known query such as “housing without a car” and ensure that relevant chunks are returned.

### 4. Retrieval

* **AI tool:** ChatGPT / Groq LLM.
* **Input:**

  * The evaluation questions listed in this planning document.
  * Retrieval requirement of Top-k = 10.
  * Vector store schema and any metadata filters.
* **Expected output:**

  * A `retrieve(query)` function that:

    * Embeds the query using `nomic-embed-text-v1_5` with search-query semantics.
    * Performs similarity search in ChromaDB with `k=10`.
    * Returns ranked chunks along with metadata such as source and URL.
* **Verification:**

  * Run retrieval for each evaluation question.
  * Inspect returned chunks to ensure they contain relevant information about neighborhoods, apartment complexes, transportation, budgets, and lease advice.
  * Adjust chunking or retrieval settings if irrelevant results appear frequently.

### 5. Generation 

* **AI tool:** Groq LLM (`llama-3.3-70b-versatile`) for answer generation; ChatGPT may be used for prompt design and refinement.
* **Input:**

  * A RAG prompt template containing:

    * The user question.
    * The top 10 retrieved chunks.
    * Source labels and URLs.
    * Instructions to answer only from the provided context, cite sources, and acknowledge uncertainty when evidence is limited.
* **Expected output:**

  * A function `answer_question(query)` that:

    * Calls `retrieve(query)`.
    * Formats the retrieved context into the prompt.
    * Sends the prompt to the Groq LLM API.
    * Returns a final markdown response with source attributions.
* **Verification:**

  * Run all five evaluation questions end-to-end.
  * Compare generated answers against the expected answers defined in the Evaluation Plan.
  * Verify that important neighborhoods, apartment complexes, transportation options, and lease considerations are correctly identified.
  * Check that claims are supported by retrieved sources and that unsupported information is clearly marked as uncertain.
  * Refine prompts and retrieval settings until answers consistently meet expectations.

