# nornickel2024
Nornickel's Hackathon: Intellectual Horizons 2024.
Tasks:

1. Time Flotation Machine

In this task, we aim to optimize ore extraction from pulp by finding the most efficient ranges under given constraints. Participants will have to consider various features, their boundaries, and the width of the ranges. Using algorithms and models, teams will be able to improve the efficiency of the enrichment process, which will lead to significant economic and environmental benefits.

Solution:
Changes:

- Using different optimization algorithms: Added method parameter to optimize_ore_extraction function, which allows to choose optimization algorithm from minimize, differential_evolution, basinhopping, dual_annealing. In main function, optimization with each of the methods is run to compare results.
- Using machine learning methods: Added predict_optimal_ranges and main functions to train a machine learning model (linear regression in this case) and predict optimal ranges based on new data.
- Accounting for process dynamics: Added a plan to account for process dynamics in the comments:
- Generate new pulp data with changed characteristics.
- Update the machine learning model using the new data.
- Predict new optimal ranges using the updated model.

Important:

- This code is still a basic example and requires further development to solve a real ore extraction optimization problem.
- It is necessary to take into account the specifics of a particular beneficiation process and use real pulp data and constraints.
 - The optimization results must be verified and validated using real experiments.
- To solve problems with process dynamics, it is necessary to develop algorithms for adapting the machine learning model to changes in pulp characteristics.
- To create a full-fledged enrichment process optimization system, it will be necessary to develop a user interface, integrate with real data collection and enrichment process management systems, as well as continuous monitoring and analysis of the results.

2. Dirty Deeds

Dirty camera lenses can seriously affect the performance of computer vision algorithms. In this task, it is necessary to develop a method for determining the degree of frame contamination to ensure reliable operation of cameras in production, as well as in robot couriers and autonomous vehicles. Solving this problem will help improve the accuracy and efficiency of various systems that depend on visual data.

Changes:

• get_pollution_level (Python) and getPollutionLevel (Arduino): Now returns an additional mask (pixel array) with the highlighted pollution areas.
• Python: 
* Use cv2.bitwise_or to combine the current frame and the pollution mask.
* Display the result using cv2.imshow.
• Arduino:
* Create a pollutionMask to store the highlighted pollution pixels.
* Use the memset function to initialize the mask to zeros.
* In the for loop in the getPollutionLevel function, when a pollution pixel is detected, the value in the mask is set to 255.
* Draw a frame to the display using the drawBitmap function and then draw black pixels in those places where the value in the pollutionMask is 255.

Result:

Now the current frame will be displayed on the screen with the highlighted pollution areas in black.

3. Multimodal RAG Models

The task is to create an efficient pipeline for automatic document search and indexing using multimodal RAG and the ColPali model. Participants will have to develop a system that can process various data formats, providing fast and accurate indexing. This solution will significantly simplify access to information and increase the productivity of working with documents in various fields.

Solution:
Description:

1. Load documents:
- load_and_preprocess_documents: Loads documents from different formats and splits the text into chunks using RecursiveCharacterTextSplitter.
2. Create vector store:
- create_vector_store: Creates a vector store using FAISS and OpenAIEmbeddings.
3. Create retrieval chain:
- create_retrieval_qa_chain: Creates a retrieval chain using RAG that includes a vector store and a language model.
4. Refine query with ColPali:
- query_with_colpali: Uses ColPali to refine the query before using RAG.
5. Main code:
- main: Starts the pipeline, loads documents, creates a vector store, creates a retrieval chain, queries for information, and outputs the answer.

 How to use:

1. Install libraries:
- pip install langchain openai faiss colpali
2. Create ColPali file: Install colpali and copy colpali.py from repository.
3. Add API keys:

- Get API key from OpenAI and add it to ~/.env file (e.g. OPENAI_API_KEY=your_key).
4. Create document1.pdf, document2.txt, document3.docx file: Add documents in different formats to be indexed in code folder.
5. Run code:

- Run main.py file.

More info:

- Error handling: Add error handling to code to improve pipeline robustness.
- Parameters: Tune parameters of models (temperature for LLM, number of nearest neighbors to search) to achieve optimal accuracy.
 - Scaling: Consider using other vector stores (e.g. Chroma, Pinecone) to handle larger data sets.
- LLM models: Experiment with different LLM models (e.g. GPT-3, GPT-4, Jurassic-1 Jumbo) to achieve the best results.

Important:

- Make sure you have internet access as the pipeline uses the OpenAI API.
- Make sure you have access to the files in the code folder (e.g. document1.pdf, document2.txt, document3.docx).

This code provides an example pipeline for automatic document search and indexing using RAG and ColPali. It can be used as a starting point for developing more complex and scalable systems.
