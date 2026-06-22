# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

This system covers **off-campus housing for University at Buffalo (UB) students**, especially new and international students looking near North and South Campus. It answers questions about which neighborhoods and apartment complexes students recommend, realistic rent ranges, commuting without a car, and what to check before signing a lease.

This knowledge is valuable because a bad choice of area, landlord, or lease can mean long commutes, safety problems, and money you can't get back once you've signed. It's hard to find through official channels: the useful specifics — which complexes have unresponsive management, real per-bedroom rents, which streets students actually live on, how to get around without a car — live scattered across Reddit threads and Facebook groups. UB's official pages only give generic safety and leasing tips.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | r/UBreddit — "best off campus housing" | Reddit thread (saved `.json`) | https://www.reddit.com/r/UBreddit/comments/158idir/best_off_campus_housing/ |
| 2 | r/UBreddit — "Hey incoming freshman looking for off campus housing" | Reddit thread (saved `.json`) | https://www.reddit.com/r/UBreddit/comments/1r9493z/hey_incoming_freshman_looking_for_off_campus/ |
| 3 | r/UBreddit — "Off-Campus Housing near North Campus" | Reddit thread (saved `.json`) | https://www.reddit.com/r/UBreddit/comments/vjvutj/offcampus_housing_near_north_campus/ |
| 4 | r/UBreddit — "Popular Off Campus Housing Neighborhoods/Streets" | Reddit thread (saved `.json`) | https://www.reddit.com/r/UBreddit/comments/ibn2fg/popular_off_campus_housing_neighborhoodsstreets/ |
| 5 | r/UBreddit — "Off Campus Housing; NO CAR" | Reddit thread (saved `.json`) | https://www.reddit.com/r/UBreddit/comments/gu82ij/off_campus_housing_no_car/ |
| 6 | UB — Off-Campus Living Guide | Official UB web page (fetched) | https://www.buffalo.edu/community/neighbors/students/off-campus-living-guide.html |
| 7 | UB — Living Off-Campus | Official UB web page (fetched) | https://www.buffalo.edu/community/information-for-students/living-off-campus.html |
| 8 | UB Off-Campus Housing Portal (OCH101) | Housing portal (fetched) | https://buffalo.och101.com |
| 9 | RentCollegePads — UB off-campus search | Listing site (fetched) | https://www.rentcollegepads.com/off-campus-housing/university-at-buffalo/search |
| 10 | Apartments.com — houses near SUNY Buffalo North Campus | Listing site (saved `.html`) | https://www.apartments.com/off-campus-housing/ny/buffalo/state-university-of-new-york-buffalo-north-campus/houses/ |

> **Ingestion note:** Sources 6–9 were fetched live by `ingest.py`. Sources 1–5 (Reddit) and 10 (Apartments.com) block automated requests, so they were saved manually in the browser and loaded from `documents/`. Reddit pages were saved as their `.json` form, which the pipeline parses into post + full comment tree. The three listing sites (8, 9, 10) are JavaScript apps that expose little static text, so most retrievable knowledge comes from the Reddit threads and UB guides.

---

## Chunking Strategy

**Chunk size:** 250 tokens (target maximum), measured with the `bert-base-uncased` WordPiece tokenizer (the same tokenizer family the embedding model `all-MiniLM-L6-v2` uses).

**Overlap:** 50 tokens (~20% of chunk size).

**Why these choices fit your documents:** I originally planned 800–1,000 tokens (see `planning.md`), but the actual corpus is small — ~9,200 tokens total. Most documents are short: a Reddit question plus a handful of comments, or a brief UB guide page. At 800–1,000 tokens nearly every document became a single chunk (16 chunks total, below the 50 floor), so a query about one apartment complex couldn't be told apart from the rest of the thread. Dropping to 250 tokens isolates individual recommendations and comments as standalone, retrievable thoughts; the 50-token overlap keeps a recommendation that straddles a boundary intact in one chunk. **Preprocessing before chunking:** for HTML sources I stripped `<script>/<style>/<nav>/<header>/<footer>` and elements whose class/id marks navigation, ads, cookie banners, and share/"read more" widgets, then unescaped HTML entities (`&amp;`, `&nbsp;`) and dropped short boilerplate lines; for Reddit I parsed the thread `.json` directly into "post + comment tree" text (no markup at all). Chunking is paragraph/sentence-aware, falling back to token windows only for oversized units, so chunks don't cut mid-sentence.

**Final chunk count:** 52 chunks across 10 documents (within the healthy 50–2,000 range).

---

## Sample Chunks

Five representative chunks produced by `ingest.py` (250-token target, 50-token overlap). Each
was chosen because it stands on its own — a reader could answer a real question from the chunk
alone — and together they span different subtopics (complex comparisons, budgets, no-car commuting,
neighborhoods, and lease advice) and both source types (Reddit threads and the UB guide).

**Chunk 1 — Source: `03_reddit_north_campus.json` (r/UBreddit — Off-Campus Housing near North Campus)**

> Post title: Off-Campus Housing near North Campus
>
> Original post by Temporary_Wafer_5756: I am currently looking for off-campus housing near North Campus.
>
> I am debating between The Station, Villas on Rensch, and University Village at Sweethome.
>
> Based on all the reviews I have read online, The Station seems to be the best choice among these. I saw some reviews online stating it doesn't have very responsive management, poor construction, and spotty wifi. I am currently considering a 4BHK apartment (the only available option) at the station. The station is expensive, so if anybody who has lived there or presently living there can let me know more about the place and their living experience? I don't want to regret my choice later.
>
> I am also open to places near the north campus or other student housing like Axis360.
>
> Comment by [deleted]: Uhhh, if you're looking for fall most have a waitlist and most aren't even accepting new people. Rensch has no spots I know for a fact

**Chunk 2 — Source: `02_reddit_incoming_freshman.json` (r/UBreddit — incoming freshman off campus)**

> Safe areas around **South Campus**: University Heights and streets within a 5–10 minute walk of Main Street and South Campus are popular. Great for big groups of friends, and you can find some 1 or 2 bedroom apartments.
>
> The typical rent is around $550–$750 per bedroom if you're budget conscious, fully furnished, and with utilities included. Cheaper options exist, but quality and responsiveness can vary.
>
> If your looking for any of the following, South is the way to go: Walkable to campus / Utilities included (makes budgeting easier) / Furnished or semi-furnished bedrooms / Group-friendly layouts
>
> If you're budget-focused, locking in a roommate group early is key. The good houses near South Campus often go fast.
>
> If you're aiming for **North Campus**, complexes like The Station at Buffalo or Villas on Rensch exist, but you'll usually pay a premium. Rent there typically runs $700–$1000+ per bedroom depending on the unit.

**Chunk 3 — Source: `05_reddit_no_car.json` (r/UBreddit — Off Campus Housing; NO CAR)**

> Comment by Eudaimonics: **1. Neighborhoods** — For that much, you can pretty much afford to live anywhere in the city.
>
> If you don't have a car, I would suggest living off South campus, West/North of Main Street. You should easily be able to find a nice 2 bedroom apartment for around $800.
>
> South is nice because you can actually walk to stores, the grocery market and restaurants. The area surrounding North is very suburban and public transportation isn't that great if you need to get out and do errands. There's also very few apartment buildings within walking distance.
>
> **2. Areas to Avoid** — The area directly South of South Campus is fine if you stick within a few blocks of Main Street. However there are a lot more slum lords and parties to deal with. The farther South you go from South Campus, the dicier it gets.
>
> **3. Landlords** — Most landlords in Buffalo just own a handful of properties, so it's kind of hard to give recommendations.

**Chunk 4 — Source: `04_reddit_neighborhoods_streets.json` (r/UBreddit — popular neighborhoods/streets)**

> Post title: Popular Off Campus Housing Neighborhoods/Streets
>
> Original post by buffalopv: I'm trying to find out what popular student neighborhoods there are off campus around North or South campus. I don't want to be in the expensive apartments near North so I'd appreciate if anyone could list streets/neighborhoods with lots of student housing on them.
>
> Comment by coconutinmee: I'm writing about South Campus here: Near Maynard — Bailey Avenue; Near Goodyear — Callodine Avenue, Springville Avenue; Near Main Circle — Englewood, Heath, Merrimac, Minesotta, Winspear Avenue. Plenty of student housing in these areas and they're close to dozens of eateries; groceries are close and convenient as well. Rent is MAAXXX $450 ish (around $320 on an average).
>
> Comment by Rookkas: i've lived on both montrose and winspear near south. the latter is the exact opposite of the other. on montrose i lived next to an old lady and a family. super chill. although a frat was down the street and regularly got busted by the cops.

**Chunk 5 — Source: `07_ub_living_off_campus.html` (UB — Living Off-Campus, official guide)**

> If the landlord refuses to answer your questions or is evasive, you may want to look for a different apartment. If a landlord agrees to make improvements, make sure you get it in writing, along with a specific timeline for completion.
>
> Visit the apartment at different times of the day and evening, to make sure it is well-lit and safe. If you are renting from out of town, schedule an appointment to see it when you arrive in Buffalo. Signing a lease without seeing the property yourself is not recommended.
>
> Remember, you may be friendly with your landlord, but that doesn't mean your landlord is your friend. Research their reputation before you sign your lease.
>
> Questions to Ask Your Potential Landlord — Appliances: Are appliances included and in good repair (stove, refrigerator, microwave, dishwasher)? Is there laundry on-site? Is it coin-operated? Utilities: Are utilities covered in rent?

**Total chunk count: 52 chunks across 10 documents** (within the healthy 50–2,000 range).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` (sentence-transformers, 384-dimensional). It runs locally with no API key and no rate limits, which suits a small corpus (52 chunks) that I re-embed repeatedly while iterating. Chunks are embedded with normalized vectors and stored in a persistent ChromaDB collection using cosine distance, with each chunk's source document name, original URL, and chunk position saved as metadata for later attribution. *(This is a change from my original plan of `nomic-embed-text-v1.5` via Groq — see planning.md → Retrieval Approach for the rationale.)*

**Production tradeoff reflection:** all-MiniLM-L6-v2 is the right call for this small, English-only, locally-iterated project. If I were deploying for real users and cost weren't a constraint, I'd weigh: **accuracy** — a larger model (e.g. `nomic-embed-text-v1.5` or a domain-tuned model) would better capture slang, abbreviations, and implicit preferences in student-written housing posts; **context length** — MiniLM truncates at 256 tokens, so longer chunks lose their tail; a long-context model would let me embed bigger chunks without aggressive splitting; **multilingual support** — international students may mix languages when searching, which MiniLM handles poorly; **latency/scale** — larger or API-hosted models add per-call latency, bigger vectors, and storage cost. I'd prioritize accuracy and multilingual robustness while balancing vector size and response time.

---

## Retrieval Test Results

Retrieved with `all-MiniLM-L6-v2` + ChromaDB (cosine distance, lower = more relevant), top-k = 5. Showing the top 3 per query.

**Query 1: "As a student without a car, which neighborhoods near UB are best for off-campus housing?"**

Top returned chunks:
- `0.272` — r/UBreddit — incoming freshman off campus (chunk #0): incoming freshman asking for safe neighborhoods, affordable shared apartments, and usual rent range near UB.
- `0.347` — r/UBreddit — best off campus housing (chunk #0): request for off-campus options near North Campus with reviews (Auden, Air Buffalo, etc.).
- `0.362` — r/UBreddit — off campus housing, NO CAR (chunk #5): comparison of living on North vs. South campus and named apartment options (Sweet Home, Chestnut, 2091).

Relevance explanation: All three come from the right sources (Reddit housing threads) and directly address neighborhood choice for students without a car; the #3 result is from the dedicated "NO CAR" thread and names concrete complexes, exactly on-topic. Distances are low (0.27–0.36), well under the 0.6–0.7 weak-match threshold.

---

**Query 2: "What do students think about popular housing complexes near North Campus?"**

Top returned chunks:
- `0.301` — r/UBreddit — incoming freshman off campus (chunk #0): freshman asking for recommendations on complexes/neighborhoods.
- `0.321` — r/UBreddit — best off campus housing (chunk #2): student warns that student-advertised complexes are pricey (furnished, shuttle, amenities) and suggests cheaper alternatives.
- `0.360` — r/UBreddit — best off campus housing (chunk #4): names specific complexes — Amherst Manor (cheap, safe, walkable to North), Block 20, Alexander Estates.

Relevance explanation: The #2 and #3 chunks are genuine opinions about specific North-Campus complexes with pros/cons (price, walkability, amenities) — precisely what the query asks. Low distances (0.30–0.36) and correct sourcing from the complex-discussion threads.

---

**Query 3: "What should I check before signing an off-campus lease?"**

Top returned chunks:
- `0.375` — UB — Off-Campus Living Guide (chunk #0): "Things to Know Before Leasing" — read the lease, tour the property, beware of scams, security deposits, know your rights.
- `0.412` — UB — Living Off-Campus (chunk #5): roommate financial/legal liability and signing a roommate contract.
- `0.492` — r/UBreddit — off campus housing, NO CAR (chunk #2): student advice to read the lease for anything sketchy and check tenant rights.

Relevance explanation: The top result is the official UB pre-lease checklist — the ideal source for this question — and the others add roommate-liability and peer advice. This query also shows good source *diversity*: the system pulled the authoritative UB guide first, then a community perspective, which is exactly the blend this project aims for.

---

## Grounded Generation

**System prompt grounding instruction:** The generator (`llama-3.3-70b-versatile` via Groq, `temperature=0`) receives a system prompt that *enforces* grounding rather than suggesting it. The numbered rules are: (1) answer **only** using the CONTEXT documents in the user message; (2) if the answer is not in the CONTEXT, reply with the exact sentence `"I don't have enough information on that."` and nothing else; (3) do not guess, infer beyond the text, or add general advice unsupported by the CONTEXT; (4) cite sources inline using the `(source: …)` labels printed above each document. The retrieved chunks are formatted as numbered, source-labeled blocks (`Document N (source: …)`), and `temperature=0` keeps the model faithful to the provided text. Because the model is told to emit one fixed refusal string, the out-of-scope path is detectable and testable.

**How source attribution is surfaced in the response:** Two ways, for safety. (1) The model cites `(source: …)` inline in its answer. (2) More importantly, attribution is **programmatic**: `ask()` returns `result["sources"]`, a deduplicated list of the `{source, url}` metadata of the actually-retrieved chunks — built in code, not by the model, so it can't be hallucinated. When the model returns the refusal string, the source list is suppressed (empty) so no misleading citations are attached to a non-answer.

---

## Example Responses

**Grounded response 1**

Query: What do students think about popular housing complexes near North Campus?

Response: Students have mixed opinions about popular housing complexes near North Campus. Some options mentioned include Amherst Manor Apartments, which is considered "cheap and safe" and within walking distance to North Campus (source: r/UBreddit — best off campus housing). Other options mentioned are Alexander Estates, Block 20, Auden Buffalo, Air Buffalo, American Campus (Villas at Rensch Chestnut or Sweethome), the Station, Creekview Court, and Colonie Apartments. However, there is a negative review of Auden Buffalo and a strong warning against Air Buffalo. Villas at Rensch and the Station seem to be well-regarded options (source: r/UBreddit — best off campus housing).

Source attribution: r/UBreddit — incoming freshman off campus; r/UBreddit — best off campus housing; r/UBreddit — popular neighborhoods/streets. *(Every claim is traceable to the retrieved chunks — the named complexes and the Auden/Air Buffalo warnings come directly from the "best off campus housing" thread.)*

---

**Grounded response 2**

Query: Is a budget of $750 per month enough for off-campus housing near UB?

Response: A budget of $750 per month can be sufficient for off-campus housing near UB, but it depends on location. For South Campus, typical rent is around $550–$750 per bedroom if you're budget conscious, fully furnished, and with utilities included (source: r/UBreddit — incoming freshman off campus). For North Campus, rent typically runs $700–$1000+ per bedroom depending on the unit (source: r/UBreddit — incoming freshman off campus). So $750 might be enough for a budget option near South Campus, but tight for North Campus.

Source attribution: r/UBreddit — incoming freshman off campus; UB — Living Off-Campus; UB Off-Campus Housing Portal (OCH101); r/UBreddit — best off campus housing.

---

**Out-of-scope query**

Query: What is the best pizza topping in Buffalo?

System response (refusal): I don't have enough information on that. *(No sources attached. The five retrieved chunks all had cosine distances of 0.68–0.72 — above the weak-match threshold — and contained no pizza content, so the grounding instruction correctly produced a refusal instead of a plausible-sounding guess from training data.)*

---

## Query Interface

**Input fields:** A single **"Your question"** text box (submit with the **Ask** button or by pressing Enter). A row of example questions is provided so a viewer can run the system without typing. Built with Gradio (`app.py`), served at `http://localhost:7860` via `python app.py`.

**Output format:** Two read-only text boxes — **"Answer"** (the grounded, citation-bearing response) and **"Retrieved from"** (a bulleted list of the source documents and their URLs, populated programmatically from retrieval metadata).

---

**Sample Interaction Transcript**

> **User:** How do students commute to North Campus without a car?

> **System (Answer):** According to the documents, students can commute to North Campus without a car by using public transportation, including buses and the UB shuttle (source: UB — Off-Campus Living Guide). Some students mention that buses for city routes come every 10–20 minutes (source: r/UBreddit — off campus housing, NO CAR), and there are bus stops near North Campus.

> **System (Retrieved from):**
> • UB — Off-Campus Living Guide — https://www.buffalo.edu/community/neighbors/students/off-campus-living-guide.html
> • r/UBreddit — off campus housing, NO CAR — https://www.reddit.com/r/UBreddit/comments/gu82ij/off_campus_housing_no_car/

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | As a student without a car, which neighborhoods near UB are best? | University Heights / Amherst / North Campus areas; transportation options; affordability | Lists University Heights and South Campus streets (Bailey, Callodine, Springville, Winspear, etc.), notes they're walkable to bus stops (best dist. 0.272) | Relevant | **Partially accurate** — strong on neighborhoods + transit access, but leans South Campus and doesn't give affordability figures or the North/Amherst angle the expected answer wanted |
| 2 | What do students think about popular complexes near North Campus? | Compare ≥2 complexes (Station, Villas, Amherst Manor) with pros/cons | Compares Amherst Manor (cheap/safe/walkable), Villas at Rensch and The Station (well-regarded), with a negative Auden review and a warning against Air Buffalo (best dist. 0.301) | Relevant | **Accurate** — multiple named complexes with real pros/cons drawn directly from the threads |
| 3 | Is $750/month enough for off-campus housing near UB? | Explain what's realistic at that budget; suggest where to search | Says $750 works for budget-conscious South Campus (~$550–750/bedroom, furnished, utilities incl.) but is tight for North ($700–1000+) (best dist. 0.372) | Relevant | **Partially accurate** — nails the realistic-budget analysis but never suggests where to search for listings |
| 4 | What should I check before signing an off-campus lease? | Review lease, check utilities, visit property, understand deposits, read reviews | Full checklist: read lease, tour property, scams, liability, security deposits, know rights, renters insurance, inspect/document damage, repairs in writing, ask about utilities (best dist. 0.375) | Relevant | **Accurate** — covers all expected points except "read reviews" of the landlord |
| 5 | How do students commute to North Campus without a car? | UB Stampede buses, apartment shuttles, walking, winter-weather challenges | Mentions public buses (10–20 min city routes), UB buses/shuttles — but omits walking, the Stampede shuttle, and winter challenges (best dist. 0.465 — the highest of all 5) | Partially relevant | **Partially accurate** — correct but thin; see Failure Case Analysis |

**Summary:** 2 Accurate, 3 Partially accurate, 0 Inaccurate. No hallucinations or ungrounded answers — the partial scores are from *missing* detail the corpus doesn't contain well, not from wrong information.

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** "How do students commute to North Campus without a car?" (eval Q5)

**What the system returned:** A correct but thin answer — it mentioned public buses (10–20 minute city routes) and "UB buses and shuttles," but omitted walking, the UB **Stampede** shuttle by name, apartment shuttles, and the winter-weather challenges the expected answer called for. Its retrieval distances were the worst of all five queries (best 0.465, others 0.47–0.55), and the #1 retrieved chunk was a generic "consider the cost, safety, commute, parking" passage rather than anything specifically about commuting.

**Root cause (tied to a specific pipeline stage):** This is an **ingestion-stage** failure, not a generation failure. On the UB Off-Campus Living Guide, the entire Transportation section is rendered as a list of *hyperlink labels* — literally `Bus fares and passes / Bus stop locations / Metro Rail / UB buses and shuttles / UB Bikeshare` — with the real content (routes, fares, the Stampede schedule, winter tips) living on **linked pages that `ingest.py` never followed** (it ingests one page per URL, not outbound links). So the chunk that *should* answer this question contains keywords with almost no semantic substance, and it even bleeds into unrelated "Code of Student Conduct" text. Because the chunk is so thin, `all-MiniLM-L6-v2` produced a weak embedding for it (little meaning per token), which is why it ranked only 4th at distance 0.519, below a loosely-related chunk that merely co-mentions the word "commute." Generation then did exactly what it should — it stayed grounded and didn't invent Stampede/winter details that simply aren't in the corpus — so the thinness traces all the way back to what was ingested.

**What you would change to fix it:** (1) In ingestion, follow one level of in-domain outbound links from the UB guide (or add the UB Stampede/transportation detail pages directly to the sources list) so the actual commuting content enters the corpus. (2) Drop near-empty, link-label-only chunks during cleaning (e.g., discard chunks below a token-substance threshold or with a high ratio of short list lines) so they don't occupy a top-k slot. (3) Add one Reddit thread specifically about commuting/winter to strengthen the community angle.

---

## Spec Reflection

**One way the spec helped you during implementation:** The Documents table and Architecture diagram in `planning.md` made the ingestion stage concrete before I wrote any code. Because each of the 10 sources was already named with a URL and a type, I could build `ingest.py` around a fixed source list with per-source handling (fetch vs. load-from-disk) instead of discovering the corpus ad hoc. The diagram's labeled stages (Ingestion → Chunking → Embedding → Vector Store → Retrieval → Generation) also gave me a clean file boundary per stage (`ingest.py`, `embed.py`, `retrieve.py`, `query.py`, `app.py`), which kept each piece independently testable. The Chunking Strategy section, even though I later changed the numbers, forced me to think about *why* a size fit the documents rather than picking one arbitrarily.

**One way your implementation diverged from the spec, and why:** Two planned numbers changed once I had real data. First, **chunk size dropped from 800–1,000 tokens to 250** — the actual corpus is only ~9,200 tokens, so large chunks collapsed each short Reddit thread into a single chunk and produced just 16 chunks total (below the 50 floor), destroying retrieval granularity. Second, the **embedding model changed from `nomic-embed-text-v1.5` (Groq API) to `all-MiniLM-L6-v2` (local) and top-k from 10 to 5** — running locally removed the need for an API key and rate limits while iterating, and with only 52 chunks, k=10 pulled in loosely related material that diluted the context. I updated `planning.md`'s Chunking Strategy, Retrieval Approach, and the diagram to record both changes and their reasoning, treating the spec as a living document rather than a fixed contract.

---

## AI Usage

**Instance 1 — Ingestion & cleaning the Reddit sources**

- *What I gave the AI:* My `planning.md` Documents table and Architecture diagram, and the instruction to build an ingestion script that loads each source, cleans boilerplate, and saves raw text before cleaning.
- *What it produced:* `ingest.py` with a BeautifulSoup-based HTML cleaner (strip `<script>/<nav>/<footer>`, remove ad/cookie/share nodes, unescape entities) and a fetch-or-load-local loader.
- *What I changed or overrode:* When the saved Reddit pages cleaned to **empty text**, we discovered Reddit renders comments in shadow DOM that "Save As" can't capture. I directed the AI to switch the Reddit path to parsing the thread's `.json` endpoint instead; it added a `parse_reddit_json()` comment-tree walker. I also had it add a safety guard so the boilerplate filter never deletes a node containing substantial text (it had been over-deleting the post body). I had to save the `.json` files manually in my browser because the server-side fetch was 403-blocked.

**Instance 2 — Chunking size (overrode my own spec)**

- *What I gave the AI:* My Chunking Strategy section specifying 800–1,000 token chunks with 150–200 overlap, and the instruction to implement token-based `chunk_text()`.
- *What it produced:* A paragraph/sentence-aware `chunk_text()` using the `bert-base-uncased` tokenizer at my specified 900/175 size.
- *What I changed or overrode:* At that size the corpus produced only **16 chunks** (under the 50 floor) because my documents turned out far shorter than I'd assumed. I had the AI run an empirical size sweep, then dropped to **250/50** (52 chunks) and updated `planning.md` with the reason — overriding my own original plan number based on real data.

**Instance 3 — Resolving a dependency conflict**

- *What I gave the AI:* The instruction to add the Gradio interface and the resulting import error.
- *What it produced:* A diagnosis that `gradio>=6.9` requires `huggingface-hub>=1.2`, which conflicted with the pinned `transformers 4.57`/`sentence-transformers 3.4.1` (capped at hub `<1.0`).
- *What I changed or overrode:* Rather than downgrade Gradio (the spec requires ≥6.9), I directed it to upgrade `transformers`/`sentence-transformers` to 5.x, confirmed the embedding model was unchanged, and had it rebuild the Chroma index and pin the new versions in `requirements.txt`.
