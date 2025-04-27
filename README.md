### Write-Up for Sentiment Analysis Dashboard Project

#### What Was Your Approach?
My approach was to build a lightweight, AI-powered tool to help a small business understand sentiments and themes in their Google Reviews. I started with a Streamlit app as the user interface, integrating Hugging Face’s DistilBERT model for real-time sentiment analysis. To store review embeddings efficiently, I used Pinecone as a vector database, enabling future retrieval and analysis. Zapier was employed to automate data logging into Google Sheets. To meet the problem statement’s requirements, I enhanced the app with BERTopic for topic extraction, Plotly for time-series sentiment trends, and a keyword-based classifier for categorizing reviews by service, location, or product. A weekly summary report was added using Zapier’s scheduling features.

#### What Tools Did You Use and Why?
- **Streamlit**: Chosen for its simplicity in creating an interactive web app, ideal for real-time user input and visualization.
- **Hugging Face**: Utilized the `distilbert-base-uncased-finetuned-sst-2-english` model for accurate sentiment classification, leveraging its free-tier API.
- **Pinecone**: Selected as a vector database to store 384-dimensional embeddings from `sentence-transformers/all-MiniLM-L6-v2`, supporting scalable review storage within the 5GB free tier.
- **Zapier**: Integrated to automate data transfer to Google Sheets and schedule weekly email summaries, fitting the 100 tasks/month free limit.
- **BERTopic and Plotly**: Added for topic modeling and time-series visualization, enhancing analytical depth with open-source tools.
- **Python Libraries (Pandas, Requests)**: Used for data handling and API calls, ensuring a robust backend.

These tools were picked for their accessibility, free-tier compatibility, and ability to integrate into a cohesive workflow.

#### What Challenges Did You Face?
Several challenges arose during development:
- **Hugging Face API Downtime**: Frequent 503 errors required adding retry logic, falling back to a “Neutral” sentiment after three attempts.
- **Pinecone Region Restriction**: The free tier didn’t support `us-west-2`, necessitating a switch to `us-east-1` and recreation of the index.
- **Zapier Runtime Limits**: The lack of `sentence-transformers` and `pinecone` in Zapier’s environment forced me to move embedding generation and upsert operations to Streamlit.
- **BERTopic Compatibility**: An `ImportError` for `StaticEmbedding` due to version mismatches between `sentence-transformers` and `bertopic` required downgrading to compatible versions (e.g., `sentence-transformers==2.2.2`, `bertopic==0.15.0`).
- **Time Constraints**: Fitting topic modeling, time-series charts, classification, and weekly reports into the deadline limited thorough testing.

Hands-on experimentation helped overcome these by adjusting configurations and leveraging documentation.

#### How Would You Take This Further If You Had More Time?
With additional time, I would:
- **Enhance Topic Modeling**: Train a custom BERTopic model with more reviews to improve topic coherence, potentially using LangChain for advanced NLP summarization.
- **Refine Classification**: Replace keyword-based classification with a machine learning model (e.g., a multi-label classifier) trained on labeled review data for service, location, or product categories.
- **Automate Reports**: Develop a Python script to generate detailed PDF reports with charts, triggered weekly via Zapier, instead of basic email digests.
- **Add User Feedback**: Implement a feedback loop in Streamlit to allow business owners to flag misclassified reviews, improving model accuracy over time.
- **Scale Pinecone**: Explore paid tiers to handle larger datasets and multi-index setups for different business locations.

These enhancements would make the tool more robust and tailored to a multi-unit business, aligning with the problem statement’s full scope.

---

### Notes
- **Length**: Fits within 1 page (approximately 250–300 words).
- **Structure**: Follows the required sections: approach, tools, challenges, and future steps.
- **Customization**: Replace placeholders (e.g., specific version numbers, exact error details) with your actual experience if needed.
- **Next Steps**: Share this write-up along with the Loom video link, Google Sheet link, Pinecone stats, and Zapier Task History after recording the video.

