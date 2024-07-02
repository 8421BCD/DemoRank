DOC_FORMAT_DIC = {
    "msmarco-v1-passage": "{contents}",
    "beir-v1.0.0-scifact-flat": "{title}. {text}",
    "beir-v1.0.0-fiqa-flat": "{text}",
    "beir-v1.0.0-nfcorpus-flat": "{title}. {text}",
    "beir-v1.0.0-fever-flat": "{title}. {text}",
    "beir-v1.0.0-climate-fever-flat": "{title}. {text}",
    "beir-v1.0.0-hotpotqa-flat": "{title}. {text}",
    "beir-v1.0.0-nq-flat": "{title}. {text}",
    "beir-v1.0.0-quora-flat": "{text}",
    "beir-v1.0.0-trec-covid-flat": "{title}. {text}",
    "beir-v1.0.0-webis-touche2020-flat": "{title}. {text}",
    "beir-v1.0.0-arguana-flat": "{title}. {text}",
    "beir-v1.0.0-dbpedia-entity-flat": "{title}. {text}",
    "beir-v1.0.0-robust04-flat": "{text}",
    "beir-v1.0.0-scidocs-flat": "{title}. {text}",
    "beir-v1.0.0-trec-news-flat": "{title}. {text}",
    "beir-v1.0.0-signal1m-flat": "{text}",
    # "beir-v1.0.0-hotpotqa-flat": "title: {title} content: {text}",
}

yes_no_system_instruction = {
    "msmarco-v1-passage": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No. If the passage is relevant to the query, output Yes; otherwise, output No.",
    ########## Bio-Medical IR ##########
    # "beir-v1.0.0-nfcorpus-flat": "Given a bio-medical passage and a query, predict whether the bio-medical passage is relevant to the query by outputting either Yes or No. If the bio-medical passage is relevant to the query, output Yes; otherwise, output No.",
    
    ########## QA ##########
    "beir-v1.0.0-nq-flat": "Given a passage and a question, predict whether the passage is relevant to the question by outputting either Yes or No. If the passage is relevant to the question, output Yes; otherwise, output No.",
    "beir-v1.0.0-hotpotqa-flat": "Given a passage and a question, predict whether the passage is relevant to the question by outputting either Yes or No. If the passage is relevant to the question, output Yes; otherwise, output No.",
    
    ########## Duplicate-Question Retrieval ##########
    # "beir-v1.0.0-quora-flat": "Given Question1 and Question2, determine whether these two questions are duplicates of each other by outputting either Yes or No. If the questions are duplicates, output Yes; otherwise, output No.",
    
    ########## Fact Checking ##########
    # "beir-v1.0.0-scifact-flat": "Given an article and a claim, predict whether the article is relevant to the claim by outputting either Yes or No. If the article is relevant to the claim, output Yes; otherwise, output No.",
    "beir-v1.0.0-fever-flat": "Given an article and a claim, predict whether the article is relevant to the claim by outputting either Yes or No. If the article is relevant to the claim, output Yes; otherwise, output No.",
    # "beir-v1.0.0-climate-fever-flat": "Given an article and a claim, predict whether the article is relevant to the claim by outputting either Yes or No. If the article is relevant to the claim, output Yes; otherwise, output No.",


    ########## others ##########
    "beir-v1.0.0-arguana-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No. If the passage is relevant to the query, output Yes; otherwise, output No.",
    "beir-v1.0.0-webis-touche2020-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No. If the passage is relevant to the query, output Yes; otherwise, output No.",
    # "beir-v1.0.0-scidocs-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No.",
    "beir-v1.0.0-scidocs-flat": "Predict whether a passage is relevant to a query.",
    "beir-v1.0.0-trec-news-flat": "Given a passage(a news article) and a query(a news headline), predict whether the passage(a news article) is relevant to the query(a news headline) by outputting either Yes or No. If the passage(a news article) is relevant to the query(a news headline), output Yes; otherwise, output No.",
    "beir-v1.0.0-robust04-flat": "Given a passage(a news article) and a query(a news query), predict whether the passage(a news article) is relevant to the query(a news query) by outputting either Yes or No. If the passage(a news article) is relevant to the query(a news query), output Yes; otherwise, output No.",
    "beir-v1.0.0-fiqa-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No.",
    "beir-v1.0.0-signal1m-flat": "Given a Twitter passage and a query, predict whether the Twitter passage is relevant to the query by outputting either Yes or No.",
    "beir-v1.0.0-trec-covid-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No. If the passage is relevant to the query, output Yes; otherwise, output No.",
    "beir-v1.0.0-nfcorpus-flat": "Given a medical passage and a query, predict whether the medical passage is relevant to the query by outputting either Yes or No. If the medical passage is relevant to the query, output Yes; otherwise, output No.",
    "beir-v1.0.0-dbpedia-entity-flat": "Given a passage(a DBPedia article) and a query(an entity-based query), predict whether the passage(a DBPedia article) is relevant to the query(an entity-based query) by outputting either Yes or No. If the passage(a DBPedia article) is relevant to the query(an entity-based query), output Yes; otherwise, output No.",
    # "beir-v1.0.0-dbpedia-entity-flat": "Predict whether a passage(a DBPedia article) is relevant to a query(an entity-based query) by outputting either Yes or No. If the passage(a DBPedia article) is relevant to the query(an entity-based query), output Yes; otherwise, output No.",
    "beir-v1.0.0-quora-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No.",
    "beir-v1.0.0-scifact-flat": "Given a passage(a scientific article) and a query(a scientific claim), output whether the passage is relevant to the query with Yes or No.",
    "beir-v1.0.0-climate-fever-flat": "Given an article passage and a claim query, predict whether the article passage is relevant to the claim query by outputting either Yes or No. If the article passage is relevant to the claim query, output Yes; otherwise, output No.",
    # "beir-v1.0.0-nq-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No. If the passage is relevant to the query, output Yes; otherwise, output No.",
    # "beir-v1.0.0-hotpotqa-flat": "Given a passage and a query, predict whether the passage is relevant to the query by outputting either Yes or No. If the passage is relevant to the query, output Yes; otherwise, output No.",

}

# "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nIs the Passage relevant to the Query?\nAnswer: ",

PROMPT_DICT_YES_NO = {
    "msmarco-v1-passage": {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "vicuna-7b-v1.5": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "Meta-Llama-3-8B-Instruct": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    
    ########## Bio-Medical IR ##########
    # 'beir-v1.0.0-nfcorpus-flat': {
    #     "flan-t5-xl": "Bio-Medical Passage: {doc}\nQuery: {qry}\nIs the Bio-Medical Passage relevant to the Query?\nOutput: ",
    #     "flan-t5-xxl": "Bio-Medical Passage: {doc}\nQuery: {qry}\nIs the Bio-Medical Passage relevant to the Query?\nOutput: ",
    #     # "flan-t5-xl": "Is the Medical Passage: {doc} relevant to the Query: {qry}?\nOutput: ",
    #     # "flan-t5-xxl": "Is the Medical Passage: {doc} relevant to the Query: {qry}?\nOutput: ",
    # },

    ########## QA ##########
    'beir-v1.0.0-nq-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
        "vicuna-7b-v1.5": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
        "Meta-Llama-3-8B-Instruct": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
    },
    'beir-v1.0.0-hotpotqa-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
        "vicuna-7b-v1.5": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
        "Meta-Llama-3-8B-Instruct": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
    },
    # 'beir-v1.0.0-fiqa-flat': {
    #     # "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nIs the Passage relevant to the Query?\nOutput: ",
    #     # "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nIs the Passage relevant to the Query?\nOutput: ",
    #     "flan-t5-xl": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
    #     "flan-t5-xxl": "Passage: {doc}\nQuestion: {qry}\nIs the Passage relevant to the Question?\nOutput: ",
    # },

    ########## Duplicate-Question Retrieval ########## OK!
    # 'beir-v1.0.0-quora-flat': {
    #     # "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    #     # "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    #     "flan-t5-xl": "Question1: {doc}\nQuestion2: {qry}\nOutput: ",
    #     "flan-t5-xxl": "Question1: {doc}\nQuestion2: {qry}\nOutput: ",
    # },

    ########## Fact Checking ##########
    # 'beir-v1.0.0-scifact-flat': {
    #     # "flan-t5-xl": "Is the Scientific Passage: {doc} relevant to the Query: {qry}?\nOutput: ",
    #     # "flan-t5-xxl": "Is the Scientific Passage: {doc} relevant to the Query: {qry}?\nOutput: ",
    #     "flan-t5-xl": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
    #     "flan-t5-xxl": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
    # },
    'beir-v1.0.0-fever-flat': {
        # "flan-t5-xl": "Article: {doc}\nMulti-hop Query: {qry}\nIs the Article relevant to the Multi-hop Query?\nOutput: ",
        # "flan-t5-xxl": "Article: {doc}\nMulti-hop Query: {qry}\nIs the Article relevant to the Multi-hop Query?\nOutput: ",
        "flan-t5-xl": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
        "flan-t5-xxl": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
        "vicuna-7b-v1.5": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
        "Meta-Llama-3-8B-Instruct": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
    },
    # 'beir-v1.0.0-climate-fever-flat': {
    #     # "flan-t5-xl": "Article: {doc}\nMulti-hop Query: {qry}\nIs the Article relevant to the Multi-hop Query?\nOutput: ",
    #     # "flan-t5-xxl": "Article: {doc}\nMulti-hop Query: {qry}\nIs the Article relevant to the Multi-hop Query?\nOutput: ",
    #     "flan-t5-xl": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
    #     "flan-t5-xxl": "Article: {doc}\nClaim: {qry}\nIs the Article relevant to the Claim?\nOutput: ",
    # },


    ############## others ##############
    'beir-v1.0.0-arguana-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-webis-touche2020-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-scidocs-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-trec-news-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-robust04-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-fiqa-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-signal1m-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-trec-covid-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-nfcorpus-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-dbpedia-entity-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-quora-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-scifact-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    'beir-v1.0.0-climate-fever-flat': {
        "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
        "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    },
    # 'beir-v1.0.0-nq-flat': {
    #     "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    #     "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    # },
    # 'beir-v1.0.0-hotpotqa-flat': {
    #     "flan-t5-xl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    #     "flan-t5-xxl": "Passage: {doc}\nQuery: {qry}\nOutput: ",
    # },
}




MSMARCO_PROMPT = """Generate a question that is the most relevant to the given document.
The document: How much does a Economist make? The average Economist salary is $103,124. Filter by location to see Economist salaries in your area. Salary estimates are based on 1,655 salaries submitted anonymously to Glassdoor by Economist employees.
Here is a generated relevant question: economics average salary

Generate a question that is the most relevant to the given document.
The document: Phoenix: Annual Weather Averages. July is the hottest month in Phoenix with an average temperature of 33Â°C (91Â°F) and the coldest is January at 12Â°C (54Â°F) with the most daily sunshine hours at 14 in June. The wettest month is August with an average of 32mm of rain.Loading weather data.uly is the hottest month in Phoenix with an average temperature of 33Â°C (91Â°F) and the coldest is January at 12Â°C (54Â°F) with the most daily sunshine hours at 14 in June. The wettest month is August with an average of 32mm of rain. Loading weather data.
Here is a generated relevant question: average temperature in phoenix in july

Generate a question that is the most relevant to the given document.
The document: Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and symptoms of Ehlers-Danlos syndrome, which vary from mildly loose joints to life-threatening complications.
Here is a generated relevant question: what is eds?

Generate a question that is the most relevant to the given document.
The document: Posted: Friday, October 23, 2015 12:00 am. Michael Coard | 1 comment. Glenn Fordâs case is nothing special and, at the same time, is very special. Itâs nothing special because it involves the same old story of racism in Americaâs legal system. Itâs also very special because it involves racism so egregious that even the white legal system has conceded it.
Here is a generated relevant question: was the actor glenn ford a racist

"""

GBQ_PROMPT = """Generate a good and a bad question to for the following documents.
Example 1:
Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.     
Good Question: How much caffeine is ok for a pregnant woman to have?
Bad Question: Is a little caffeine ok during pregnancy?

Example 2:
Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.
Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?
Bad Question: What fruit is native to Australia?

Example 3:
Document: The Canadian Armed Forces. 1  The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2  There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3  In Canada, August 9 is designated as National Peacekeepers' Day.
Good Question: Information on the Canadian Armed Forces size and history.
Bad Question: How large is the Canadian military?

Example 4:
Document: {doc}"""

DEFAULT_PROMPT = "Generate a question that is the most relevant to the given document." \
                 "\nThe document: {doc}\nHere is a generated relevant question: "

stablelm_system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
alpaca_system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
                       "Write a response that appropriately completes the request.\n\n"
PROMPT_DICT = {
    "msmarco-v1-passage": {
        "llama-2-7b": "Generate a question that is the most relevant to the given passage."
                                       "\nPassage: {doc}\n\nHere is a generated relevant question: ",
        "llama-2-13b": "Generate a question that is the most relevant to the given passage."
                                       "\nPassage: {doc}\n\nHere is a generated relevant question: ",
        "llama-2-7b-chat-hf": "Generate a question that is the most relevant to the given passage."
                                       "\nPassage: {doc}\n\nHere is a generated relevant question: ",
        "yulan-chat-2-13b": "Generate a question that is the most relevant to the given passage."
                                       "\nPassage: {doc}\n\nHere is a generated relevant question: ",
        "flan-t5-xl": "Generate a question that is the most relevant to the given passage."
                                       "\nPassage: {doc}\n\nHere is a generated relevant question: ",
        "flan-t5-xxl": "Generate a question that is the most relevant to the given passage."
                                       "\nPassage: {doc}\n\nHere is a generated relevant question: ",
    },
    "beir-v1.0.0-trec-covid-flat": {
        "flan-t5-xl": "Generate a question that is the most relevant to the given article's title and abstract."
                             "\n{doc}",
        "flan-t5-xxl": "Generate a question that is the most relevant to the given article's title and abstract."
                             "\n{doc}",

        "bigscience/T0_3B": "Please write a question based on this passage.\n{doc}",

        "castorini/doc2query-t5-large-msmarco": "{doc}",

        "llama-2-7b": "Generate a question that is the most relevant to the given article's title and abstract."
                               "\n{doc}\n\nHere is a generated relevant question: ",

        "huggyllama/llama-13b": "Generate a question that is the most relevant to the given article's title and abstract."
                                "\n{doc}\n\nHere is a generated relevant question: ",

        "stanford_alpaca": alpaca_system_prompt + "### Instruction:\nGenerate a question that is the most relevant to the given article's title and abstract.\n\n"
                                                  "### Input:\n{doc}\n\n### Response:",

        'TheBloke/stable-vicuna-13B-HF': "### Human: Generate a question that is the most relevant to the given article's title and abstract."
                                         "\n{doc}\n### Assistant: Here is a generated relevant question: ",

        "tiiuae/falcon-7b-instruct": "Generate a question that is the most relevant to the given article's title and abstract."
                                     "\n{doc}\nHere is a generated relevant question: ",

        'tiiuae/falcon-40b-instruct': "Generate a question that is the most relevant to the given article's title and abstract."
                                      "\n{doc}\nHere is a generated relevant question: ",

        "stabilityai/stablelm-tuned-alpha-7b": stablelm_system_prompt + "<|USER|>Generate a question that is the most relevant to the given article's title and abstract."
                                                                        "\n{doc}<|ASSISTANT|>Here is a generated relevant question: ",
        # "stabilityai/stablelm-tuned-alpha-7b": stablelm_system_prompt + "<|USER|>Is the given question relevant to the given article's title and abstract."
        #                                                                 "\nThe question: {qry}\nThe artical: {doc}<|ASSISTANT|>The answer is: ",
    },

    "beir-v1.0.0-dbpedia-entity-flat": {
        "flan-t5-xl": "Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                             "\n{doc}",
        "flan-t5-xxl": "Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                             "\n{doc}",

        "bigscience/T0_3B": "Please write a question based on this passage.\n{doc}",

        "castorini/doc2query-t5-large-msmarco": "{doc}",

        "llama-2-7b": "Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                               "\n{doc}\n\nHere is a generated relevant query that includes an entity: ",

        "huggyllama/llama-13b": "Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                                "\n{doc}\n\nHere is a generated relevant query that includes an entity: ",

        "stanford_alpaca": alpaca_system_prompt + "### Instruction:\nGenerate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract.\n\n"
                                                  "### Input:\n{doc}\n\n### Response:",

        'TheBloke/stable-vicuna-13B-HF': "### Human: Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                                         "\n{doc}\n### Assistant: Here is a generated relevant query that includes an entity: ",

        "tiiuae/falcon-7b-instruct": "Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                                     "\n{doc}\nHere is a generated relevant query that includes an entity: ",

        "tiiuae/falcon-40b-instruct": "Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                                      "\n{doc}\nHere is a generated relevant query that includes an entity: ",

        "stabilityai/stablelm-tuned-alpha-7b": stablelm_system_prompt + "<|USER|>Generate a query that includes an entity and is also highly relevant to the given Wikipedia page title and abstract."
                                                                        "\n{doc}<|ASSISTANT|>Here is a generated relevant query that includes an entity: ",
    },

    'beir-v1.0.0-robust04-flat': {
        "flan-t5-xl": "Generate a question that is the most relevant to the given document."
                             "\n{doc}",

        "bigscience/T0_3B": "Please write a question based on this passage.\n{doc}",

        "castorini/doc2query-t5-large-msmarco": "{doc}",

        "llama-2-7b": "Generate a question that is the most relevant to the given document."
                               "\nThe document: {doc}\nHere is a generated relevant question: ",

        "huggyllama/llama-13b": "Generate a question that is the most relevant to the given document."
                                "\nThe document: {doc}\nHere is a generated relevant question: ",

        "stanford_alpaca": alpaca_system_prompt + "### Instruction:\nGenerate a question that is the most relevant to the given document.\n\n"
                                                  "### Input:\n{doc}\n\n### Response:",

        "tiiuae/falcon-7b-instruct": "Generate a question that is the most relevant to the given document."
                                     "\nThe document: {doc}\nHere is a generated relevant question: ",

        "tiiuae/falcon-40b-instruct": "Generate a question that is the most relevant to the given document."
                                      "\nThe document: {doc}\nHere is a generated relevant question: ",

        'TheBloke/stable-vicuna-13B-HF': "### Human: Generate a question that is the most relevant to the given document."
                                         "\nThe document: {doc}\n### Assistant: Here is a generated relevant question: ",

        "stabilityai/stablelm-tuned-alpha-7b": stablelm_system_prompt + "<|USER|>Generate a question that is the most relevant to the given document."
                                                                        "\nThe document: {doc}<|ASSISTANT|>Here is a generated relevant question: ",
    },
    'beir-v1.0.0-fiqa-flat': {
        "flan-t5-xl": "Generate a question that is the most relevant to the given document."
                             "\n{doc}",

        "bigscience/T0_3B": "Please write a question based on this passage.\n{doc}",

        "castorini/doc2query-t5-large-msmarco": "{doc}",

        "tiiuae/falcon-7b-instruct": "Generate a question that is the most relevant to the given document."
                                     "\nThe document: {doc}\nHere is a generated relevant question: ",

        "tiiuae/falcon-40b-instruct": "Generate a question that is the most relevant to the given document."
                                      "\nThe document: {doc}\nHere is a generated relevant question: ",

        "llama-2-7b": "Generate a question that is the most relevant to the given document."
                               "\nThe document: {doc}\nHere is a generated relevant question: ",

        "huggyllama/llama-13b": "Generate a question that is the most relevant to the given document."
                                "\nThe document: {doc}\nHere is a generated relevant question: ",

        "stanford_alpaca": alpaca_system_prompt + "### Instruction:\nGenerate a question that is the most relevant to the given document.\n\n"
                                                  "### Input:\n{doc}\n\n### Response:",

        'TheBloke/stable-vicuna-13B-HF': "### Human: Generate a question that is the most relevant to the given document."
                                         "\nThe document: {doc}\n### Assistant: Here is a generated relevant question: ",

        "stabilityai/stablelm-tuned-alpha-7b": stablelm_system_prompt + "<|USER|>Generate a question that is the most relevant to the given document."
                                                                        "\nThe document: {doc}<|ASSISTANT|>Here is a generated relevant question: ",
    },
}

# prompt having chain-of-thought examplers, for LLaMA
MSMARCO_PROMPT_COT = """
Document: How much does a Economist make? The average Economist salary is $103,124. Filter by location to see Economist salaries in your area. Salary estimates are based on 1,655 salaries submitted anonymously to Glassdoor by Economist employees.
Query: economist's average salary
Relevance judgement: The query is asking for the average salary of an economist, and the document is about the average salary of an economist. So, the query is relevant to the document. 

Document: Phoenix: Annual Weather Averages. July is the hottest month in Phoenix with an average temperature of 33Â°C (91Â°F) and the coldest is January at 12Â°C (54Â°F) with the most daily sunshine hours at 14 in June. The wettest month is August with an average of 32mm of rain.Loading weather data.uly is the hottest month in Phoenix with an average temperature of 33Â°C (91Â°F) and the coldest is January at 12Â°C (54Â°F) with the most daily sunshine hours at 14 in June. The wettest month is August with an average of 32mm of rain. Loading weather data.
Query: average temperature in phoenix in july
Relevance judgement: The query is asking for the average temperature in Phoenix in July, and the document mentions the average temperature in Phoenix in July explicitly. So, the query is relevant to the document.

Document: Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and symptoms of Ehlers-Danlos syndrome, which vary from mildly loose joints to life-threatening complications.
Query: what is eds?
Relevance judgement: The query is asking for the definition of EDS, and the document is about the definition of EDS. So, the query is relevant to the document.

Document: Posted: Friday, October 23, 2015 12:00 am. Michael Coard | 1 comment. Glenn Fordâs case is nothing special and, at the same time, is very special. Itâs nothing special because it involves the same old story of racism in Americaâs legal system. Itâs also very special because it involves racism so egregious that even the white legal system has conceded it.
Query: was the actor glenn ford a racist
Relevance judgement: The query is asking if Glenn Ford was a racist, and the document is talking about a legal case of Glenn Forda where racism is involved very egregiously. So, the query is relevant to the document.
"""

MSMARCO_PROMPT_COT_NEG = """
Document: How much does a Economist make? The average Economist salary is $103,124. Filter by location to see Economist salaries in your area. Salary estimates are based on 1,655 salaries submitted anonymously to Glassdoor by Economist employees.
Query: The Economist
Relevance judgement: The query is looking for information about the megazine "The Economist", buth the document is about the average salary of an economist. So, the query is not relevant to the document. 

Document: Phoenix: Annual Weather Averages. July is the hottest month in Phoenix with an average temperature of 33Â°C (91Â°F) and the coldest is January at 12Â°C (54Â°F) with the most daily sunshine hours at 14 in June. The wettest month is August with an average of 32mm of rain.Loading weather data.uly is the hottest month in Phoenix with an average temperature of 33Â°C (91Â°F) and the coldest is January at 12Â°C (54Â°F) with the most daily sunshine hours at 14 in June. The wettest month is August with an average of 32mm of rain. Loading weather data.
Query: average temperature in phoenix in April
Relevance judgement: The query is asking for the average temperature in Phoenix in April. Though the document is talking the temperature in Phoenix, but it only metions that of January and July but not April. So, the query is not relevant to the document.

Document: Ehlers-Danlos syndrome is a group of disorders that affect the connective tissues that support the skin, bones, blood vessels, and many other organs and tissues. Defects in connective tissues cause the signs and symptoms of Ehlers-Danlos syndrome, which vary from mildly loose joints to life-threatening complications.
Query: treatment of eds?
Relevance judgement: The query is asking for the treatment of EDS. The document though is about EDS, it does not mention the treatment of EDS. So, the query is not relevant to the document.

Document: Posted: Friday, October 23, 2015 12:00 am. Michael Coard | 1 comment. Glenn Fordâs case is nothing special and, at the same time, is very special. Itâs nothing special because it involves the same old story of racism in Americaâs legal system. Itâs also very special because it involves racism so egregious that even the white legal system has conceded it.
Query: films starred by glenn ford
Relevance judgement: The query is asking Glenn Ford's films, and the document is talking about a legal case of Glenn Forda. So, the query is not relevant to the document.
"""
